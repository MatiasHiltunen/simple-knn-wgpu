use simple_knn_wgpu::{compute_knn, GpuContext};
use anyhow::Result;
use rand::Rng;
use std::time::Instant;

/// Naive CPU reference implementation for k=3 mean distance.
fn cpu_knn_mean_distances(points: &[f32]) -> Vec<f32> {
    let n = points.len() / 3;
    let mut dists = vec![0.0f32; n];

    for i in 0..n {
        // Extract point i
        let xi = points[i * 3];
        let yi = points[i * 3 + 1];
        let zi = points[i * 3 + 2];

        // Keep three smallest distances squared
        let mut best = [f32::INFINITY; 3];

        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = points[j * 3];
            let yj = points[j * 3 + 1];
            let zj = points[j * 3 + 2];

            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            // Insert into best three (simple insertion sort style)
            if dist_sq < best[0] {
                best[2] = best[1];
                best[1] = best[0];
                best[0] = dist_sq;
            } else if dist_sq < best[1] {
                best[2] = best[1];
                best[1] = dist_sq;
            } else if dist_sq < best[2] {
                best[2] = dist_sq;
            }
        }

        // Mean of distances (not squared)
        let mean_dist = (best[0].sqrt() + best[1].sqrt() + best[2].sqrt()) / 3.0;
        dists[i] = mean_dist;
    }

    dists
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    // Generate random point cloud
    let num_points = 200; // adjustable
    println!("Generating {} random points in unit cube...", num_points);
    let mut rng = rand::rng();
    let mut points = Vec::with_capacity(num_points * 3);
    for _ in 0..num_points {
        points.push(rng.random::<f32>()); // x
        points.push(rng.random::<f32>()); // y
        points.push(rng.random::<f32>()); // z
    }

    // CPU reference
    let t0 = Instant::now();
    let cpu_dists = cpu_knn_mean_distances(&points);
    let cpu_ms = t0.elapsed().as_secs_f32() * 1000.0;
    println!("CPU reference completed in {:.2} ms", cpu_ms);

    // GPU computation
    let gpu = GpuContext::new().await?;
    let t1 = Instant::now();
    let gpu_result = compute_knn(&gpu, &points).await?;
    let gpu_ms = t1.elapsed().as_secs_f32() * 1000.0;
    println!("GPU computation completed in {:.2} ms (device time: {:.2?} ms)", gpu_ms, gpu_result.compute_time_ms);

    // Verify correctness & print some sample values
    let mut max_abs_err = 0.0f32;
    for (i, (&cpu_val, &gpu_val)) in cpu_dists.iter().zip(gpu_result.distances.iter()).enumerate() {
        let err = (cpu_val - gpu_val).abs();
        max_abs_err = max_abs_err.max(err);

        // Print the first few points (or large errors) for inspection
        if i < 5 || err > 1e-3 {
            println!(
                "Point {:3}: CPU {:.6}  GPU {:.6}  |err| {:.6}",
                i, cpu_val, gpu_val, err
            );
        }
    }

    println!("\nMax absolute error: {:.6}", max_abs_err);
    assert!(max_abs_err < 1e-3, "GPU results deviate from CPU reference too much");

    println!("Validation passed!\n");

    Ok(())
} 