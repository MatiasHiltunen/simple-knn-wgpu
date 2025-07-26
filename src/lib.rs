//! High-performance k-nearest neighbor search using wgpu compute shaders.
//! 
//! This library provides GPU-accelerated computation of k-nearest neighbors
//! for 3D point clouds. It uses Morton encoding for spatial sorting and
//! efficient box-based partitioning to achieve near-linear performance.
//! 
//! # Example
//! 
//! ```rust,no_run
//! use simple_knn_wgpu::{compute_knn, GpuContext};
//! 
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize GPU context
//! let gpu = GpuContext::new().await?;
//! 
//! // Prepare your 3D points (flat array: x0,y0,z0,x1,y1,z1,...)
//! let points = vec![
//!     0.0, 0.0, 0.0,
//!     1.0, 0.0, 0.0,
//!     0.0, 1.0, 0.0,
//!     0.0, 0.0, 1.0,
//! ];
//! 
//! // Compute mean distances to 3 nearest neighbors
//! let result = compute_knn(&gpu, &points).await?;
//! 
//! println!("Mean distances: {:?}", result.distances);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

// Module declarations
pub mod device;
pub mod error;
pub mod knn;
pub mod morton;
pub mod shaders;
pub mod types;

// Re-exports for convenience
pub use device::GpuContext;
pub use error::{KnnError, Result};
pub use types::{KnnConfig, KnnResult, Point3};

use knn::KnnCompute;

/// Computes the mean distance to the 3 nearest neighbors for each point.
/// 
/// This is a convenience function that creates a KNN compute engine with
/// optimal configuration for your GPU and runs the computation.
/// 
/// # Arguments
/// * `context` - The GPU context to use
/// * `points` - Flat array of 3D points [x0, y0, z0, x1, y1, z1, ...]
/// 
/// # Returns
/// A `KnnResult` containing the mean distances for each point
/// 
/// # Example
/// ```rust,no_run
/// # use simple_knn_wgpu::{compute_knn, GpuContext};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let gpu = GpuContext::new().await?;
/// let points = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let result = compute_knn(&gpu, &points).await?;
/// assert_eq!(result.distances.len(), 2); // One distance per point
/// # Ok(())
/// # }
/// ```
pub async fn compute_knn(context: &GpuContext, points: &[f32]) -> Result<KnnResult> {
    let compute = KnnCompute::with_optimal_config(context.clone())?;
    compute.compute_knn(points).await
}

/// Computes KNN with a custom configuration.
/// 
/// Use this function when you need fine-grained control over the algorithm
/// parameters, such as box size or maximum points.
/// 
/// # Arguments
/// * `context` - The GPU context to use
/// * `config` - Custom KNN configuration
/// * `points` - Flat array of 3D points
/// 
/// # Returns
/// A `KnnResult` containing the mean distances for each point
pub async fn compute_knn_with_config(
    context: &GpuContext,
    config: KnnConfig,
    points: &[f32],
) -> Result<KnnResult> {
    let compute = KnnCompute::new(context.clone(), config)?;
    compute.compute_knn(points).await
}

/// Library version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point3_creation() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
        assert!(p.is_finite());
    }
    
    #[test]
    fn test_invalid_point() {
        let p = Point3::new(f32::NAN, 0.0, 0.0);
        assert!(!p.is_finite());
    }
    
    #[test]
    fn test_knn_config_validation() {
        let mut config = KnnConfig::default();
        assert!(config.validate().is_ok());
        
        config.k = 0;
        assert!(config.validate().is_err());
        
        config.k = 3;
        config.box_size = 1023; // Not power of 2
        assert!(config.validate().is_err());
    }
}
