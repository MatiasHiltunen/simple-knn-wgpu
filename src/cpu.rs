use crate::{
    error::{KnnError, Result, ValidationError},
    types::{KnnResult, Point3},
};
use rayon::prelude::*;

/// CPU fallback implementation using SIMD and rayon parallelism.
pub fn compute_knn_cpu(points: &[f32]) -> Result<KnnResult> {
    validate_points(points)?;
    let start = std::time::Instant::now();
    let n = points.len() / 3;
    let mut xs = vec![0.0f32; n];
    let mut ys = vec![0.0f32; n];
    let mut zs = vec![0.0f32; n];
    for i in 0..n {
        xs[i] = points[i * 3];
        ys[i] = points[i * 3 + 1];
        zs[i] = points[i * 3 + 2];
    }
    let distances: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| unsafe { knn_point_simd(i, &xs, &ys, &zs) })
        .collect();
    let ms = start.elapsed().as_secs_f32() * 1000.0;
    Ok(KnnResult {
        distances,
        compute_time_ms: Some(ms),
    })
}

fn validate_points(points: &[f32]) -> Result<()> {
    if points.is_empty() {
        return Err(KnnError::InvalidInput(
            ValidationError::EmptyArray.to_string(),
        ));
    }
    if points.len() % 3 != 0 {
        return Err(KnnError::InvalidInput(
            ValidationError::InvalidShape(points.len() / 3, points.len() % 3).to_string(),
        ));
    }
    for (idx, chunk) in points.chunks_exact(3).enumerate() {
        let p = Point3::new(chunk[0], chunk[1], chunk[2]);
        if !p.is_finite() {
            return Err(KnnError::InvalidInput(
                ValidationError::InvalidValues(idx).to_string(),
            ));
        }
    }
    Ok(())
}

#[inline]
unsafe fn knn_point_simd(i: usize, xs: &[f32], ys: &[f32], zs: &[f32]) -> f32 {
    let xi = xs[i];
    let yi = ys[i];
    let zi = zs[i];

    let mut best = [f32::INFINITY; 3];

    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        let xi_v = _mm_set1_ps(xi);
        let yi_v = _mm_set1_ps(yi);
        let zi_v = _mm_set1_ps(zi);
        let mut j = 0;
        while j + 4 <= xs.len() {
            let xj = _mm_loadu_ps(xs.as_ptr().add(j));
            let yj = _mm_loadu_ps(ys.as_ptr().add(j));
            let zjv = _mm_loadu_ps(zs.as_ptr().add(j));
            let dx = _mm_sub_ps(xj, xi_v);
            let dy = _mm_sub_ps(yj, yi_v);
            let dz = _mm_sub_ps(zjv, zi_v);
            let mut dist = _mm_mul_ps(dx, dx);
            dist = _mm_add_ps(dist, _mm_mul_ps(dy, dy));
            dist = _mm_add_ps(dist, _mm_mul_ps(dz, dz));
            let mut out = [0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), dist);
            for k in 0..4 {
                let idx = j + k;
                if idx == i || idx >= xs.len() {
                    continue;
                }
                let d = out[k];
                if d < best[0] {
                    best[2] = best[1];
                    best[1] = best[0];
                    best[0] = d;
                } else if d < best[1] {
                    best[2] = best[1];
                    best[1] = d;
                } else if d < best[2] {
                    best[2] = d;
                }
            }
            j += 4;
        }
        for j in j..xs.len() {
            if j == i {
                continue;
            }
            let dx = xs[j] - xi;
            let dy = ys[j] - yi;
            let dz = zs[j] - zi;
            let d = dx * dx + dy * dy + dz * dz;
            if d < best[0] {
                best[2] = best[1];
                best[1] = best[0];
                best[0] = d;
            } else if d < best[1] {
                best[2] = best[1];
                best[1] = d;
            } else if d < best[2] {
                best[2] = d;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let xi_v = vdupq_n_f32(xi);
        let yi_v = vdupq_n_f32(yi);
        let zi_v = vdupq_n_f32(zi);
        let mut j = 0;
        while j + 4 <= xs.len() {
            let xj = vld1q_f32(xs.as_ptr().add(j));
            let yj = vld1q_f32(ys.as_ptr().add(j));
            let zjv = vld1q_f32(zs.as_ptr().add(j));
            let dx = vsubq_f32(xj, xi_v);
            let dy = vsubq_f32(yj, yi_v);
            let dz = vsubq_f32(zjv, zi_v);
            let mut dist = vmulq_f32(dx, dx);
            dist = vmlaq_f32(dist, dy, dy);
            dist = vmlaq_f32(dist, dz, dz);
            let mut out = [0f32; 4];
            vst1q_f32(out.as_mut_ptr(), dist);
            for k in 0..4 {
                let idx = j + k;
                if idx == i || idx >= xs.len() {
                    continue;
                }
                let d = out[k];
                if d < best[0] {
                    best[2] = best[1];
                    best[1] = best[0];
                    best[0] = d;
                } else if d < best[1] {
                    best[2] = best[1];
                    best[1] = d;
                } else if d < best[2] {
                    best[2] = d;
                }
            }
            j += 4;
        }
        for j in j..xs.len() {
            if j == i {
                continue;
            }
            let dx = xs[j] - xi;
            let dy = ys[j] - yi;
            let dz = zs[j] - zi;
            let d = dx * dx + dy * dy + dz * dz;
            if d < best[0] {
                best[2] = best[1];
                best[1] = best[0];
                best[0] = d;
            } else if d < best[1] {
                best[2] = best[1];
                best[1] = d;
            } else if d < best[2] {
                best[2] = d;
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for (j, (&xj, (&yj, &zj))) in xs.iter().zip(ys.iter().zip(zs.iter())).enumerate() {
            if j == i {
                continue;
            }
            let dx = xj - xi;
            let dy = yj - yi;
            let dz = zj - zi;
            let d = dx * dx + dy * dy + dz * dz;
            if d < best[0] {
                best[2] = best[1];
                best[1] = best[0];
                best[0] = d;
            } else if d < best[1] {
                best[2] = best[1];
                best[1] = d;
            } else if d < best[2] {
                best[2] = d;
            }
        }
    }

    (best[0].sqrt() + best[1].sqrt() + best[2].sqrt()) / 3.0
}
