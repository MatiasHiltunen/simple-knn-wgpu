//! Basic example demonstrating simple-knn-wgpu usage.
//! 
//! This example creates a small point cloud and computes the mean distance
//! to the 3 nearest neighbors for each point.

use simple_knn_wgpu::{compute_knn, GpuContext};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("Initializing GPU context...");
    let gpu = GpuContext::new().await?;
    println!("Using GPU: {}", gpu.device_description());
    
    // Create a simple 3D point cloud
    // Points form a 2x2x2 cube
    let points = vec![
        // Bottom face
        0.0, 0.0, 0.0,  // Point 0
        1.0, 0.0, 0.0,  // Point 1
        0.0, 1.0, 0.0,  // Point 2
        1.0, 1.0, 0.0,  // Point 3
        // Top face
        0.0, 0.0, 1.0,  // Point 4
        1.0, 0.0, 1.0,  // Point 5
        0.0, 1.0, 1.0,  // Point 6
        1.0, 1.0, 1.0,  // Point 7
    ];
    
    let num_points = points.len() / 3;
    println!("\nComputing KNN for {} points...", num_points);
    
    // Compute KNN
    let result = compute_knn(&gpu, &points).await?;
    
    // Display results
    println!("\nResults:");
    println!("========");
    for (i, &distance) in result.distances.iter().enumerate() {
        println!("Point {}: mean distance to 3 nearest neighbors = {:.4}", i, distance);
    }
    
    if let Some(time) = result.compute_time_ms {
        println!("\nComputation time: {:.2} ms", time);
    }
    
    // Verify results make sense
    // In a unit cube, the minimum distance between corners is 1.0
    let min_distance = result.distances.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_distance = result.distances.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    println!("\nStatistics:");
    println!("Min mean distance: {:.4}", min_distance);
    println!("Max mean distance: {:.4}", max_distance);
    
    Ok(())
} 