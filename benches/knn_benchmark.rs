//! Benchmarks for the simple-knn-wgpu library.
//! 
//! This benchmark suite tests the performance of KNN computation
//! across different point cloud sizes and configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use simple_knn_wgpu::{compute_knn, GpuContext};
use rand::prelude::*;
use std::time::Duration;

/// Generates a random 3D point cloud.
fn generate_random_points(num_points: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(num_points * 3);
    
    for _ in 0..num_points {
        points.push(rng.gen_range(-100.0..100.0));
        points.push(rng.gen_range(-100.0..100.0));
        points.push(rng.gen_range(-100.0..100.0));
    }
    
    points
}

/// Generates points on a regular grid.
fn generate_grid_points(side_length: usize) -> Vec<f32> {
    let mut points = Vec::with_capacity(side_length * side_length * side_length * 3);
    
    for x in 0..side_length {
        for y in 0..side_length {
            for z in 0..side_length {
                points.push(x as f32);
                points.push(y as f32);
                points.push(z as f32);
            }
        }
    }
    
    points
}

fn benchmark_knn_random(c: &mut Criterion) {
    // Initialize GPU context once
    let gpu = pollster::block_on(GpuContext::new())
        .expect("Failed to initialize GPU");
    
    println!("Benchmarking on: {}", gpu.device_description());
    
    let mut group = c.benchmark_group("knn_random_points");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    // Test different point cloud sizes
    for &num_points in &[1000, 5000, 10000, 50000, 100000] {
        let points = generate_random_points(num_points, 42);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_points),
            &points,
            |b, points| {
                b.iter(|| {
                    pollster::block_on(async {
                        let result = compute_knn(&gpu, black_box(points)).await.unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_knn_grid(c: &mut Criterion) {
    let gpu = pollster::block_on(GpuContext::new())
        .expect("Failed to initialize GPU");
    
    let mut group = c.benchmark_group("knn_grid_points");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    // Test different grid sizes
    for &side in &[10, 15, 20, 30, 40] {
        let num_points = side * side * side;
        let points = generate_grid_points(side);
        
        group.bench_with_input(
            BenchmarkId::new("grid", num_points),
            &points,
            |b, points| {
                b.iter(|| {
                    pollster::block_on(async {
                        let result = compute_knn(&gpu, black_box(points)).await.unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_knn_scaling(c: &mut Criterion) {
    let gpu = pollster::block_on(GpuContext::new())
        .expect("Failed to initialize GPU");
    
    let mut group = c.benchmark_group("knn_scaling");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);
    
    // Test scaling with larger point clouds
    for &num_points in &[10_000, 50_000, 100_000, 250_000, 500_000] {
        if num_points > 100_000 {
            // Reduce sample size for very large clouds
            group.sample_size(5);
        }
        
        let points = generate_random_points(num_points, 42);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_points),
            &points,
            |b, points| {
                b.iter(|| {
                    pollster::block_on(async {
                        let result = compute_knn(&gpu, black_box(points)).await.unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_knn_random,
    benchmark_knn_grid,
    benchmark_knn_scaling
);
criterion_main!(benches); 