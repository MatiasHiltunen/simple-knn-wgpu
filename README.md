# simple-knn-wgpu

K-nearest neighbor search for 3D point clouds using WGPU compute shaders. Please notice that this project is in it's early stages and performance compared to other solutions has not been measured.

This library aims to provide cross-platform GPU acceleration for KNN computations using wgpu. It's primarily designed as an alternative to CUDA-based implementations.

Based on python package [simple-knn](https://github.com/camenduru/simple-knn).

## Features

- KNN search using compute shaders
- Spatial indexing with Morton codes (Z-order curve)
- Box-based partitioning for scalable performance
- Cross-platform support via wgpu (Vulkan, Metal, D3D12, WebGPU)
- Optimized for 3-NN (finds 3 nearest neighbors per point)


## Quick Start

```rust
use simple_knn_wgpu::{compute_knn, GpuContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU
    let gpu = GpuContext::new().await?;
    
    // Your 3D points as a flat array [x0,y0,z0,x1,y1,z1,...]
    let points = vec![
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    
    // Compute mean distances to 3 nearest neighbors
    let result = compute_knn(&gpu, &points).await?;
    
    // One distance per point
    println!("Distances: {:?}", result.distances);
    
    Ok(())
}
```

## Advanced Usage

### Custom Configuration

```rust
use simple_knn_wgpu::{compute_knn_with_config, GpuContext, KnnConfig};

let config = KnnConfig {
    k: 3,                    // Number of neighbors (currently fixed at 3)
    box_size: 512,           // Spatial partition size (must be power of 2)
    max_points: 10_000_000,  // Maximum supported points
    ..Default::default()
};

let result = compute_knn_with_config(&gpu, config, &points).await?;
```

### GPU Selection

```rust
// Use specific adapter
let instance = wgpu::Instance::default();
let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })
    .await
    .unwrap();

let gpu = GpuContext::from_adapter(adapter).await?;
```

## Algorithm Overview

1. Global Bounding Box, parallel reduction to find point cloud bounds
2. Morton Encoding for spatial indexing using Z-order curve for locality
3. Points are sorted by Morton code for spatial coherence
4. Sorted points are divided into fixed-size boxes
5. Compute AABB for each partition
6. KNN Search for each point:
   - Check immediate neighbors in sorted order (very fast)
   - Use box AABBs to cull distant partitions
   - Exhaustively search nearby boxes



## Examples

Run the basic example:

```bash
cargo run --example basic
```

## License

This project is dual-licensed under MIT OR Apache-2.0.

### Future Improvements

- [ ] GPU-based radix sort (currently using CPU sort)
- [ ] Variable K support (currently fixed at k=3)
- [ ] Half-precision float support
- [ ] WebGPU browser demo
- [ ] Benchmarks against other implementations 
