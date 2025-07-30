//! GPU device management and initialization.
//!
//! This module handles the creation and configuration of wgpu devices,
//! ensuring we select appropriate hardware and configure it optimally
//! for our KNN compute workloads.

use crate::{
    error::{KnnError, Result},
    types::KnnConfig,
};
use log::{debug, info};
use std::sync::Arc;

/// GPU device context containing all resources needed for compute operations.
#[derive(Clone)]
pub struct GpuContext {
    /// The wgpu device for creating GPU resources
    pub device: Arc<wgpu::Device>,
    /// The command queue for submitting GPU work
    pub queue: Arc<wgpu::Queue>,
    /// Information about the selected adapter
    pub adapter_info: wgpu::AdapterInfo,
    /// Configured limits for this device
    pub limits: wgpu::Limits,
    /// Supported features on this device
    pub features: wgpu::Features,
}

impl GpuContext {
    /// Creates a new GPU context with the best available adapter.
    ///
    /// This function will:
    /// 1. Enumerate all available adapters
    /// 2. Select the most powerful one (preferring discrete GPUs)
    /// 3. Create a device with appropriate limits
    ///
    /// # Errors
    /// Returns an error if no suitable GPU adapter is found or device creation fails.
    pub async fn new() -> Result<Self> {
        let instance = create_instance();
        let adapter = select_best_adapter(&instance).await?;

        Self::from_adapter(adapter).await
    }

    /// Creates a GPU context from a specific adapter.
    ///
    /// This is useful when you want to control adapter selection manually.
    pub async fn from_adapter(adapter: wgpu::Adapter) -> Result<Self> {
        let adapter_info = adapter.get_info();
        info!(
            "Selected GPU adapter: {} ({:?})",
            adapter_info.name, adapter_info.device_type
        );

        // Request device with features we need
        // Enable push constants for efficient radix sort parameters
        let required_features = wgpu::Features::PUSH_CONSTANTS;
        let required_limits = wgpu::Limits {
            max_bind_groups: 4,
            max_storage_buffer_binding_size: 128 * 1024 * 1024, // 128MB (more conservative)
            max_buffer_size: 512 * 1024 * 1024,                 // 512MB (more conservative)
            max_compute_workgroup_storage_size: 32 * 1024,      // 32KB
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("KNN Compute Device"),
                required_features,
                required_limits: required_limits.clone(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| KnnError::GpuInitError(format!("Failed to create device: {}", e)))?;

        let limits = device.limits();
        let features = device.features();

        debug!("Device limits: {:?}", limits);
        debug!("Device features: {:?}", features);

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
            features,
        })
    }

    /// Creates a KNN configuration optimized for this device.
    ///
    /// The configuration will respect device limits and choose appropriate
    /// workgroup sizes for optimal performance.
    pub fn create_optimal_config(&self) -> KnnConfig {
        // Choose box size based on device capabilities
        let max_workgroup = self.limits.max_compute_invocations_per_workgroup;
        let box_size = if max_workgroup >= 1024 {
            1024
        } else if max_workgroup >= 512 {
            512
        } else if max_workgroup >= 256 {
            256
        } else {
            128
        };

        // Calculate maximum points based on available memory
        // We need roughly 40 bytes per point (point data + morton + indices + distances)
        let max_buffer_size = self.limits.max_buffer_size as usize;
        let bytes_per_point = 40;
        let max_points = (max_buffer_size / bytes_per_point).min(100_000_000) as u32;

        KnnConfig {
            k: 3,
            box_size,
            max_points,
            max_workgroup_size: self.limits.max_compute_invocations_per_workgroup,
        }
    }

    /// Checks if the device supports the given configuration.
    pub fn supports_config(&self, config: &KnnConfig) -> Result<()> {
        config.validate().map_err(|e| KnnError::InvalidInput(e))?;

        if config.box_size > self.limits.max_compute_invocations_per_workgroup {
            return Err(KnnError::NotSupported(format!(
                "Box size {} exceeds device limit of {}",
                config.box_size, self.limits.max_compute_invocations_per_workgroup
            )));
        }

        // Check memory requirements
        let bytes_per_point = 40;
        let required_memory = config.max_points as usize * bytes_per_point;
        if required_memory > self.limits.max_buffer_size as usize {
            return Err(KnnError::NotSupported(format!(
                "Configuration requires {} bytes but device only supports {}",
                required_memory, self.limits.max_buffer_size
            )));
        }

        Ok(())
    }

    /// Gets a human-readable description of the GPU device.
    pub fn device_description(&self) -> String {
        format!(
            "{} ({:?}, driver: {})",
            self.adapter_info.name, self.adapter_info.device_type, self.adapter_info.driver
        )
    }
}

/// Creates a wgpu instance with appropriate backends for the platform.
fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    })
}

/// Selects the best available adapter for compute workloads.
async fn select_best_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
    // First try to get a high-performance (discrete) GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await;

    if let Ok(adapter) = adapter {
        let info = adapter.get_info();
        if matches!(info.device_type, wgpu::DeviceType::DiscreteGpu) {
            return Ok(adapter);
        }
    }

    // If no discrete GPU, try any available adapter
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .max_by_key(|adapter| {
            let info = adapter.get_info();
            // Prioritize by device type
            match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 3,
                wgpu::DeviceType::IntegratedGpu => 2,
                wgpu::DeviceType::VirtualGpu => 1,
                wgpu::DeviceType::Cpu => 0,
                wgpu::DeviceType::Other => 1,
            }
        });

    adapter.ok_or_else(|| KnnError::GpuInitError("No suitable GPU adapter found".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = KnnConfig {
            k: 3,
            box_size: 1024,
            max_points: 1_000_000,
            max_workgroup_size: 1024,
        };

        assert!(config.validate().is_ok());

        // Test invalid configurations
        let invalid_config = KnnConfig {
            k: 0,
            ..config.clone()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_config = KnnConfig {
            box_size: 1023, // Not power of 2
            ..config
        };
        assert!(invalid_config.validate().is_err());
    }
}
