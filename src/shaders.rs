//! Shader management module for loading and compiling WGSL shaders.
//! 
//! This module handles the loading, compilation, and caching of all
//! compute shaders used in the KNN algorithm.

use crate::error::{KnnError, Result};
use std::borrow::Cow;

/// All shader sources used in the KNN algorithm.
pub struct ShaderSources {
    /// Bounding box computation shader
    pub bbox: &'static str,
    /// Morton code computation shader
    pub morton: &'static str,
    /// Box bounding box computation shader
    pub box_bbox: &'static str,
    /// KNN search shader
    pub knn: &'static str,
}

impl Default for ShaderSources {
    fn default() -> Self {
        Self {
            bbox: include_str!("shaders/bbox.wgsl"),
            morton: include_str!("shaders/morton.wgsl"),
            box_bbox: include_str!("shaders/box_bbox.wgsl"),
            knn: include_str!("shaders/knn.wgsl"),
        }
    }
}

/// Compiled shader modules ready for use in pipelines.
pub struct CompiledShaders {
    /// Bounding box computation shader module
    pub bbox: wgpu::ShaderModule,
    /// Morton code computation shader module
    pub morton: wgpu::ShaderModule,
    /// Box bounding box computation shader module
    pub box_bbox: wgpu::ShaderModule,
    /// KNN search shader module
    pub knn: wgpu::ShaderModule,
}

impl CompiledShaders {
    /// Compiles all shaders for the given device.
    /// 
    /// # Errors
    /// Returns an error if any shader fails to compile.
    pub fn compile(device: &wgpu::Device, sources: &ShaderSources) -> Result<Self> {
        let bbox = compile_shader(device, "bbox", sources.bbox)?;
        let morton = compile_shader(device, "morton", sources.morton)?;
        let box_bbox = compile_shader(device, "box_bbox", sources.box_bbox)?;
        let knn = compile_shader(device, "knn", sources.knn)?;
        
        Ok(Self {
            bbox,
            morton,
            box_bbox,
            knn,
        })
    }
}

/// Compiles a single shader module.
fn compile_shader(
    device: &wgpu::Device,
    name: &str,
    source: &str,
) -> Result<wgpu::ShaderModule> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    
    Ok(module)
}

/// Validates that a shader source contains expected entry points.
pub fn validate_shader_entry_points(source: &str, expected: &[&str]) -> Result<()> {
    for entry_point in expected {
        if !source.contains(&format!("fn {}", entry_point)) {
            return Err(KnnError::ShaderError(format!(
                "Missing entry point '{}' in shader",
                entry_point
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shader_sources_load() {
        let sources = ShaderSources::default();
        
        // Verify all shaders are non-empty
        assert!(!sources.bbox.is_empty());
        assert!(!sources.morton.is_empty());
        assert!(!sources.box_bbox.is_empty());
        assert!(!sources.knn.is_empty());
        
        // Verify entry points exist
        assert!(validate_shader_entry_points(
            sources.bbox,
            &["compute_bbox", "reduce_bbox"]
        ).is_ok());
        
        assert!(validate_shader_entry_points(
            sources.morton,
            &["compute_morton"]
        ).is_ok());
        
        assert!(validate_shader_entry_points(
            sources.box_bbox,
            &["compute_box_bbox"]
        ).is_ok());
        
        assert!(validate_shader_entry_points(
            sources.knn,
            &["compute_knn"]
        ).is_ok());
    }
} 