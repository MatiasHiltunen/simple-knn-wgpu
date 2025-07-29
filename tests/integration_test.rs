//! Integration tests for simple-knn-wgpu.
//!
//! These tests verify that the GPU implementation produces correct results
//! for various point cloud configurations.

use approx::assert_relative_eq;
use simple_knn_wgpu::{GpuContext, KnnConfig, compute_knn_auto, compute_knn_cpu};

#[tokio::test]
async fn test_simple_cube() {
    // 8 points forming a unit cube
    let points = vec![
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 2
        1.0, 1.0, 0.0, // 3
        0.0, 0.0, 1.0, // 4
        1.0, 0.0, 1.0, // 5
        0.0, 1.0, 1.0, // 6
        1.0, 1.0, 1.0, // 7
    ];

    let result = compute_knn_auto(&points).await.unwrap();

    assert_eq!(result.distances.len(), 8);

    // Each corner of a unit cube has 3 neighbors at distances:
    // - 3 edge neighbors at distance 1.0
    // So mean distance should be 1.0
    for (_i, &dist) in result.distances.iter().enumerate() {
        assert_relative_eq!(dist, 1.0, epsilon = 0.001);
    }
}

#[tokio::test]
async fn test_line_of_points() {
    // Points along a line with unit spacing
    let points = vec![
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        2.0, 0.0, 0.0, // 2
        3.0, 0.0, 0.0, // 3
        4.0, 0.0, 0.0, // 4
    ];

    let result = compute_knn_auto(&points).await.unwrap();

    assert_eq!(result.distances.len(), 5);

    // End points have neighbors at distances 1, 2, 3
    // Mean = (1 + 2 + 3) / 3 = 2.0
    assert_relative_eq!(result.distances[0], 2.0, epsilon = 0.001);
    assert_relative_eq!(result.distances[4], 2.0, epsilon = 0.001);

    // Middle points have closer neighbors
    // Point 2 has neighbors at distances 1, 1, 2
    // Mean = (1 + 1 + 2) / 3 = 1.333...
    assert_relative_eq!(result.distances[2], 4.0 / 3.0, epsilon = 0.001);
}

#[tokio::test]
async fn test_single_cluster() {
    // Dense cluster at origin with one outlier
    let mut points = Vec::new();

    // 10 points clustered near origin
    for i in 0..10 {
        let angle = (i as f32) * std::f32::consts::TAU / 10.0;
        points.push(angle.cos() * 0.1);
        points.push(angle.sin() * 0.1);
        points.push(0.0);
    }

    // One outlier far away
    points.push(10.0);
    points.push(0.0);
    points.push(0.0);

    let result = compute_knn_auto(&points).await.unwrap();

    assert_eq!(result.distances.len(), 11);

    // Points in the cluster should have small distances
    for i in 0..10 {
        assert!(
            result.distances[i] < 0.3,
            "Clustered point {} has too large distance: {}",
            i,
            result.distances[i]
        );
    }

    // The outlier should have much larger distance
    assert!(
        result.distances[10] > 5.0,
        "Outlier has too small distance: {}",
        result.distances[10]
    );
}

#[tokio::test]
async fn test_custom_config() {
    // Test with custom configuration
    let config = KnnConfig {
        k: 3,
        box_size: 256, // Smaller box size
        max_points: 1000,
        max_workgroup_size: 256,
    };

    let points = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let gpu = GpuContext::new().await.ok();
    let result = if let Some(gpu) = gpu {
        simple_knn_wgpu::compute_knn_with_config(&gpu, config, &points)
            .await
            .unwrap()
    } else {
        compute_knn_cpu(&points).unwrap()
    };

    assert_eq!(result.distances.len(), 4);

    // Verify computation time is recorded
    assert!(result.compute_time_ms.is_some());
    assert!(result.compute_time_ms.unwrap() >= 0.0);
}

#[tokio::test]
async fn test_empty_input() {
    let points = vec![];

    let result = compute_knn_auto(&points).await;

    // Should return an error for empty input
    assert!(result.is_err());
}

#[tokio::test]
async fn test_invalid_input() {
    // Not a multiple of 3
    let points = vec![1.0, 2.0];

    let result = compute_knn_auto(&points).await;

    // Should return an error for invalid input shape
    assert!(result.is_err());
}

#[tokio::test]
async fn test_nan_handling() {
    let points = vec![0.0, 0.0, 0.0, f32::NAN, 0.0, 0.0];

    let result = compute_knn_auto(&points).await;

    // Should return an error for NaN values
    assert!(result.is_err());
}
