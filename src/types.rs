//! Type definitions for GPU data structures.
//! 
//! This module defines all the data structures that are shared between
//! CPU and GPU, ensuring proper alignment and layout for GPU usage.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A 3D point representation that is GPU-compatible.
/// 
/// Uses explicit repr(C) and bytemuck traits to ensure the data layout
/// matches what the GPU expects in compute shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Point3 {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Z coordinate
    pub z: f32,
    /// Padding for 16-byte alignment (required by many GPUs)
    pub _padding: f32,
}

impl Point3 {
    /// Creates a new point with the given coordinates.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _padding: 0.0 }
    }
    
    /// Creates a point from a glam Vec3.
    #[inline]
    pub fn from_vec3(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
    
    /// Converts this point to a glam Vec3.
    #[inline]
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    
    /// Validates that the point contains finite values.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Axis-aligned bounding box for spatial partitioning.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BoundingBox {
    /// Minimum corner of the box
    pub min: Point3,
    /// Maximum corner of the box
    pub max: Point3,
}

impl BoundingBox {
    /// Creates a new bounding box from min and max points.
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }
    
    /// Creates an empty bounding box (inverted min/max).
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }
    
    /// Expands the bounding box to include the given point.
    pub fn expand(&mut self, point: &Point3) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }
    
    /// Computes the squared distance from the box to a point.
    /// Returns 0.0 if the point is inside the box.
    pub fn distance_squared_to_point(&self, point: &Point3) -> f32 {
        let dx = 0.0f32.max(self.min.x - point.x).max(point.x - self.max.x);
        let dy = 0.0f32.max(self.min.y - point.y).max(point.y - self.max.y);
        let dz = 0.0f32.max(self.min.z - point.z).max(point.z - self.max.z);
        dx * dx + dy * dy + dz * dz
    }
}

/// Configuration parameters for the KNN algorithm.
#[derive(Clone, Debug)]
pub struct KnnConfig {
    /// Number of nearest neighbors to find (currently fixed at 3).
    pub k: u32,
    /// Size of spatial partitioning boxes (must be power of 2).
    pub box_size: u32,
    /// Maximum number of points that can be processed.
    pub max_points: u32,
    /// Device limits for workgroup sizes.
    pub max_workgroup_size: u32,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self {
            k: 3,
            box_size: 256,  // Match the max_workgroup_size default
            max_points: 10_000_000, // 10M points default limit
            max_workgroup_size: 256, // Conservative default
        }
    }
}

impl KnnConfig {
    /// Validates the configuration against device limits.
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("k must be greater than 0".to_string());
        }
        if !self.box_size.is_power_of_two() {
            return Err("box_size must be a power of 2".to_string());
        }
        if self.box_size > self.max_workgroup_size {
            return Err(format!(
                "box_size ({}) exceeds max_workgroup_size ({})",
                self.box_size, self.max_workgroup_size
            ));
        }
        Ok(())
    }
    
    /// Calculates the number of boxes needed for the given number of points.
    pub fn num_boxes(&self, num_points: u32) -> u32 {
        (num_points + self.box_size - 1) / self.box_size
    }
}

/// GPU buffer binding information.
#[derive(Clone, Debug)]
pub struct BufferBinding {
    /// The buffer resource
    pub buffer: wgpu::Buffer,
    /// Offset into the buffer
    pub offset: u64,
    /// Size of the binding
    pub size: Option<u64>,
}

/// Result of the KNN computation.
#[derive(Clone, Debug)]
pub struct KnnResult {
    /// Mean distances to k nearest neighbors for each point
    pub distances: Vec<f32>,
    /// Optional timing information (in milliseconds)
    pub compute_time_ms: Option<f32>,
} 