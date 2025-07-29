//! Main K-Nearest Neighbors implementation using wgpu compute shaders.
//!
//! This module provides the core KNN algorithm implementation that
//! orchestrates multiple GPU compute passes to efficiently find the
//! k nearest neighbors for each point in a 3D point cloud.

use crate::{
    device::GpuContext,
    error::{KnnError, Result, ValidationError},
    shaders::{CompiledShaders, ShaderSources},
    types::{BoundingBox, KnnConfig, KnnResult, Point3},
};
use bytemuck::cast_slice;
use log::{debug, info};
use std::time::Instant;
use wgpu::util::DeviceExt;

/// Main KNN compute engine.
pub struct KnnCompute {
    /// GPU context
    context: GpuContext,
    /// Configuration
    config: KnnConfig,
    /// Compute pipelines
    pipelines: ComputePipelines,
}

/// All compute pipelines used in the KNN algorithm.
struct ComputePipelines {
    bbox_compute: wgpu::ComputePipeline,
    bbox_reduce: wgpu::ComputePipeline,
    morton: wgpu::ComputePipeline,
    box_bbox: wgpu::ComputePipeline,
    knn: wgpu::ComputePipeline,
    radix_count: wgpu::ComputePipeline,
    radix_reorder: wgpu::ComputePipeline,
}

impl KnnCompute {
    /// Creates a new KNN compute engine with the given context and configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid or incompatible with the device.
    pub fn new(context: GpuContext, config: KnnConfig) -> Result<Self> {
        // Validate configuration
        context.supports_config(&config)?;

        // Compile shaders
        let shader_sources = ShaderSources::default();
        let shaders = CompiledShaders::compile(&context.device, &shader_sources)?;

        // Create compute pipelines
        let pipelines = create_pipelines(&context.device, &shaders)?;

        Ok(Self {
            context,
            config,
            pipelines,
        })
    }

    /// Creates a new KNN compute engine with optimal configuration for the device.
    pub fn with_optimal_config(context: GpuContext) -> Result<Self> {
        let config = context.create_optimal_config();
        Self::new(context, config)
    }

    /// Computes the mean distance to the 3 nearest neighbors for each point.
    ///
    /// # Arguments
    /// * `points` - Flat array of 3D points [x0, y0, z0, x1, y1, z1, ...]
    ///
    /// # Returns
    /// Array of mean distances, one per point
    ///
    /// # Errors
    /// Returns an error if the input is invalid or GPU computation fails.
    pub async fn compute_knn(&self, points: &[f32]) -> Result<KnnResult> {
        let start_time = Instant::now();

        // Validate and prepare input
        let points = validate_and_prepare_points(points)?;
        let num_points = points.len() as u32;

        info!("Computing KNN for {} points", num_points);

        // Create GPU buffers
        let buffers = self.create_buffers(&points)?;

        // ------------------------------------------------------------------
        // Phase 1: Compute global bounding box + Morton codes (GPU)
        // ------------------------------------------------------------------
        let phase1_start = Instant::now();
        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("KNN Phase 1 Encoder"),
                    });

            // Pass 1: Compute global bounding box
            let global_bbox_buffer =
                self.compute_global_bbox(&mut encoder, &buffers, num_points)?;

            // Pass 2: Compute Morton codes
            self.compute_morton_codes(&mut encoder, &buffers, &global_bbox_buffer, num_points)?;

            // Submit phase 1
            self.context.queue.submit(Some(encoder.finish()));
        }
        info!(
            "Phase 1 (bbox + morton) submitted in {:.2} ms",
            phase1_start.elapsed().as_secs_f32() * 1000.0
        );
        // ------------------------------------------------------------------

        // GPU step: radix sort points by Morton code
        let sort_start = Instant::now();
        let sorted_indices = self.sort_by_morton(&buffers, num_points).await?;
        info!(
            "GPU sort by Morton completed in {:.2} ms",
            sort_start.elapsed().as_secs_f32() * 1000.0
        );

        // ------------------------------------------------------------------
        // Phase 2: Box-level bounding boxes + KNN search (GPU)
        // ------------------------------------------------------------------
        let phase2_start = Instant::now();
        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("KNN Phase 2 Encoder"),
                    });

            // Pass 3: Compute box bounding boxes
            self.compute_box_bboxes(&mut encoder, &buffers, &sorted_indices, num_points)?;

            // Pass 4: KNN search
            self.compute_knn_search(&mut encoder, &buffers, &sorted_indices, num_points)?;

            // Submit phase 2
            self.context.queue.submit(Some(encoder.finish()));
        }
        info!(
            "Phase 2 (box bbox + knn) submitted in {:.2} ms",
            phase2_start.elapsed().as_secs_f32() * 1000.0
        );

        // Read back results
        let distances = self.read_distances(&buffers.distances, num_points).await?;

        let compute_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        info!("KNN computation completed in {:.2} ms", compute_time_ms);

        Ok(KnnResult {
            distances,
            compute_time_ms: Some(compute_time_ms),
        })
    }

    /// Creates all GPU buffers needed for the computation.
    fn create_buffers(&self, points: &[Point3]) -> Result<ComputeBuffers> {
        let device = &self.context.device;
        let num_points = points.len() as u32;
        let num_boxes = self.config.num_boxes(num_points);

        // Points buffer
        let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Points Buffer"),
            contents: cast_slice(points),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Morton codes buffer
        let morton_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morton Codes Buffer"),
            size: (num_points * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Indices buffer initialized with 0..num_points
        let indices_data: Vec<u32> = (0..num_points).collect();
        let indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Indices Buffer"),
            contents: cast_slice(&indices_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let box_bboxes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Box BBoxes Buffer"),
            size: (num_boxes * std::mem::size_of::<BoundingBox>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let distances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Distances Buffer"),
            size: (num_points * 3 * std::mem::size_of::<f32>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffers with proper alignment (256 bytes minimum)
        let num_points_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Num Points Uniform"),
            size: 256, // Uniform buffers need 256 byte alignment
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write the actual data
        self.context
            .queue
            .write_buffer(&num_points_uniform, 0, bytemuck::bytes_of(&num_points));

        let config_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Uniform"),
            size: 256, // Uniform buffers need 256 byte alignment
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write the actual data
        let config_data = [num_points, self.config.box_size, num_boxes, 3u32];
        self.context
            .queue
            .write_buffer(&config_uniform, 0, bytemuck::bytes_of(&config_data));

        Ok(ComputeBuffers {
            points: points_buffer,
            morton: morton_buffer,
            indices: indices_buffer,
            box_bboxes: box_bboxes,
            distances: distances,
            num_points_uniform: num_points_uniform,
            config_uniform: config_uniform,
        })
    }

    /// Computes the global bounding box of all points.
    fn compute_global_bbox(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &ComputeBuffers,
        num_points: u32,
    ) -> Result<wgpu::Buffer> {
        debug!("Computing global bounding box");

        // First pass: compute per-workgroup bounding boxes
        let num_workgroups = num_points.div_ceil(256).max(1);

        let workgroup_boxes_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Workgroup Boxes Buffer"),
            size: (num_workgroups * std::mem::size_of::<BoundingBox>() as u32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BBox Compute Bind Group"),
                layout: &self.pipelines.bbox_compute.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: workgroup_boxes_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.num_points_uniform.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BBox Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.bbox_compute);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Create a dedicated bind group for the reduce pass (only needs the workgroup boxes)
        let reduce_bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BBox Reduce Bind Group"),
                layout: &self.pipelines.bbox_reduce.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 1,
                    resource: workgroup_boxes_buffer.as_entire_binding(),
                }],
            });

        // Second pass: reduce workgroup results
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BBox Reduce Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.bbox_reduce);
            pass.set_bind_group(0, &reduce_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Store the workgroup boxes buffer for later use
        // *buffers.workgroup_boxes.borrow_mut() = Some(workgroup_boxes_buffer);

        Ok(workgroup_boxes_buffer)
    }

    /// Computes Morton codes for all points.
    fn compute_morton_codes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &ComputeBuffers,
        global_bbox_buffer: &wgpu::Buffer,
        num_points: u32,
    ) -> Result<()> {
        debug!("Computing Morton codes");

        // let global_bbox_buffer = buffers.workgroup_boxes.borrow()
        //     .as_ref()
        //     .ok_or_else(|| KnnError::ComputeError("Missing global bbox buffer".to_string()))?
        //     .clone();

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Morton Bind Group"),
                layout: &self.pipelines.morton.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: global_bbox_buffer,
                            offset: 0,
                            size: std::num::NonZero::new(std::mem::size_of::<BoundingBox>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.morton.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.num_points_uniform.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Morton Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.morton);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_points.div_ceil(256), 1, 1);
        }

        Ok(())
    }

    /// Sorts points by Morton code using a GPU radix sort.
    async fn sort_by_morton(
        &self,
        buffers: &ComputeBuffers,
        num_points: u32,
    ) -> Result<wgpu::Buffer> {
        debug!("Sorting by Morton code (GPU radix sort)");

        let device = &self.context.device;
        let temp_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Indices"),
            size: (num_points * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let temp_morton = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp Morton"),
            size: (num_points * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prefix = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Prefix"),
            size: (num_points * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Counts"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let workgroups = num_points.div_ceil(256);
        let mut src_keys = &buffers.morton;
        let mut dst_keys = &temp_morton;
        let mut src_indices = &buffers.indices;
        let mut dst_indices = &temp_indices;

        for bit in 0..30u32 {
            self.context
                .queue
                .write_buffer(&counts, 0, bytemuck::bytes_of(&[0u32, 0u32]));
            self.context
                .queue
                .write_buffer(&params, 0, bytemuck::bytes_of(&[num_points, bit]));

            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Radix Count BG"),
                layout: &self.pipelines.radix_count.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_keys.as_entire_binding(),
                    },
                    // wgpu::BindGroupEntry {
                    //     binding: 1,
                    //     resource: src_indices.as_entire_binding(),
                    // },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prefix.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radix Count Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.radix_count);
                pass.set_bind_group(0, &bg1, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.context.queue.submit(Some(encoder.finish()));

            self.context
                .queue
                .write_buffer(&params, 0, bytemuck::bytes_of(&[num_points, bit]));
            let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Radix Reorder BG"),
                layout: &self.pipelines.radix_reorder.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: prefix.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dst_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: dst_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: params.as_entire_binding(),
                    },
                ],
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Radix Reorder Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.radix_reorder);
                pass.set_bind_group(0, &bg2, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.context.queue.submit(Some(encoder.finish()));

            std::mem::swap(&mut src_keys, &mut dst_keys);
            std::mem::swap(&mut src_indices, &mut dst_indices);
        }

        Ok(src_indices.clone())
    }

    /// Computes bounding boxes for each spatial partition box.
    fn compute_box_bboxes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &ComputeBuffers,
        sorted_indices: &wgpu::Buffer,
        num_points: u32,
    ) -> Result<()> {
        debug!("Computing box bounding boxes");

        let num_boxes = self.config.num_boxes(num_points);

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Box BBox Bind Group"),
                layout: &self.pipelines.box_bbox.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sorted_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.box_bboxes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.config_uniform.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Box BBox Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.box_bbox);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_boxes, 1, 1);
        }

        Ok(())
    }

    /// Performs the KNN search for all points.
    fn compute_knn_search(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &ComputeBuffers,
        sorted_indices: &wgpu::Buffer,
        num_points: u32,
    ) -> Result<()> {
        debug!("Computing KNN search");

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("KNN Bind Group"),
                layout: &self.pipelines.knn.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.points.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sorted_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.box_bboxes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.distances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.config_uniform.as_entire_binding(),
                    },
                ],
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("KNN Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.knn);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_points.div_ceil(256), 1, 1);
        }

        Ok(())
    }

    /// Reads the computed distances from GPU memory.
    async fn read_distances(
        &self,
        distances_buffer: &wgpu::Buffer,
        num_points: u32,
    ) -> Result<Vec<f32>> {
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Distances Staging Buffer"),
            size: (num_points * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            distances_buffer,
            0,
            &staging_buffer,
            0,
            (num_points * 4) as u64,
        );
        self.context.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.context
            .device
            .poll(wgpu::PollType::Wait)
            .map_err(|_| KnnError::ComputeError("Failed to poll device".to_string()))?;
        rx.await.unwrap().map_err(|e| KnnError::BufferMapError(e))?;

        let data = buffer_slice.get_mapped_range();
        let distances: Vec<f32> = cast_slice(&data).to_vec();

        Ok(distances)
    }
}

/// GPU buffers used in the computation.
struct ComputeBuffers {
    points: wgpu::Buffer,
    morton: wgpu::Buffer,
    indices: wgpu::Buffer,
    box_bboxes: wgpu::Buffer,
    distances: wgpu::Buffer,
    num_points_uniform: wgpu::Buffer,
    config_uniform: wgpu::Buffer,
}

/// Creates all compute pipelines.
fn create_pipelines(device: &wgpu::Device, shaders: &CompiledShaders) -> Result<ComputePipelines> {
    // Bounding box compute pipeline
    let bbox_compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BBox Compute Pipeline"),
        layout: None,
        module: &shaders.bbox,
        entry_point: Some("compute_bbox"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Bounding box reduce pipeline
    let bbox_reduce = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BBox Reduce Pipeline"),
        layout: None,
        module: &shaders.bbox,
        entry_point: Some("reduce_bbox"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Morton code pipeline
    let morton = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Morton Pipeline"),
        layout: None,
        module: &shaders.morton,
        entry_point: Some("compute_morton"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Box bounding box pipeline
    let box_bbox = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Box BBox Pipeline"),
        layout: None,
        module: &shaders.box_bbox,
        entry_point: Some("compute_box_bbox"),
        compilation_options: Default::default(),
        cache: None,
    });

    // KNN search pipeline
    let knn = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("KNN Pipeline"),
        layout: None,
        module: &shaders.knn,
        entry_point: Some("compute_knn"),
        compilation_options: Default::default(),
        cache: None,
    });

    let radix_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Radix Count Pipeline"),
        layout: None,
        module: &shaders.radix,
        entry_point: Some("radix_count"),
        compilation_options: Default::default(),
        cache: None,
    });

    let radix_reorder = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Radix Reorder Pipeline"),
        layout: None,
        module: &shaders.radix,
        entry_point: Some("radix_reorder"),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(ComputePipelines {
        bbox_compute,
        bbox_reduce,
        morton,
        box_bbox,
        knn,
        radix_count,
        radix_reorder,
    })
}

/// Validates and prepares input points.
fn validate_and_prepare_points(points: &[f32]) -> Result<Vec<Point3>> {
    if points.is_empty() {
        return Err(ValidationError::EmptyArray.into());
    }

    if points.len() % 3 != 0 {
        return Err(ValidationError::InvalidShape(points.len() / 3, points.len() % 3).into());
    }

    let num_points = points.len() / 3;
    let mut result = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let x = points[i * 3];
        let y = points[i * 3 + 1];
        let z = points[i * 3 + 2];

        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(ValidationError::InvalidValues(i).into());
        }

        result.push(Point3::new(x, y, z));
    }

    Ok(result)
}
