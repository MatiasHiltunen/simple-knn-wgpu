//! Morton code (Z-order curve) utilities for spatial indexing.
//! 
//! Morton codes interleave the bits of 3D coordinates to create a
//! space-filling curve that preserves spatial locality. This is essential
//! for efficient spatial partitioning in our KNN algorithm.

use crate::types::Point3;

/// Prepares a 10-bit coordinate value for Morton encoding.
/// 
/// This function spreads the bits of a 10-bit value to prepare for
/// interleaving with other coordinates. The algorithm works by:
/// 1. Spreading bits to leave gaps for other coordinates
/// 2. Using bit masks to preserve only the desired bits at each step
/// 
/// # Example
/// ```ignore
/// // Input:  0000000000 0000000000 00XXXXXXXXXX (10 bits)
/// // Output: 00X00X00X00X00X00X00X00X00X (30 bits with gaps)
/// ```
#[inline]
pub fn prepare_morton_coordinate(x: u32) -> u32 {
    debug_assert!(x < 1024, "Coordinate must be less than 1024 (10 bits)");
    
    let mut x = x;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Encodes a 3D point into a 30-bit Morton code.
/// 
/// The Morton code interleaves the bits of the x, y, and z coordinates
/// to create a single integer that preserves spatial locality. Points
/// that are close in 3D space tend to have similar Morton codes.
/// 
/// # Arguments
/// * `point` - The 3D point to encode
/// * `min` - Minimum corner of the bounding box
/// * `max` - Maximum corner of the bounding box
/// 
/// # Returns
/// A 30-bit Morton code (using 10 bits per coordinate)
pub fn encode_morton(point: &Point3, min: &Point3, max: &Point3) -> u32 {
    // Normalize coordinates to [0, 1] range
    let norm_x = (point.x - min.x) / (max.x - min.x);
    let norm_y = (point.y - min.y) / (max.y - min.y);
    let norm_z = (point.z - min.z) / (max.z - min.z);
    
    // Scale to 10-bit integer range [0, 1023]
    let scale = ((1u32 << 10) - 1) as f32;
    let x = (norm_x.clamp(0.0, 1.0) * scale) as u32;
    let y = (norm_y.clamp(0.0, 1.0) * scale) as u32;
    let z = (norm_z.clamp(0.0, 1.0) * scale) as u32;
    
    // Prepare each coordinate for interleaving
    let x = prepare_morton_coordinate(x);
    let y = prepare_morton_coordinate(y);
    let z = prepare_morton_coordinate(z);
    
    // Interleave: x gets bits 0, 3, 6, ...; y gets 1, 4, 7, ...; z gets 2, 5, 8, ...
    x | (y << 1) | (z << 2)
}

/// Batch encodes multiple points into Morton codes.
/// 
/// This is more efficient than encoding points one by one when processing
/// large datasets, as it allows for better CPU cache utilization.
pub fn batch_encode_morton(
    points: &[Point3],
    min: &Point3,
    max: &Point3,
    output: &mut [u32],
) {
    assert_eq!(
        points.len(),
        output.len(),
        "Points and output arrays must have the same length"
    );
    
    // Pre-calculate scaling factors to avoid repeated division
    let scale = ((1u32 << 10) - 1) as f32;
    let inv_range_x = 1.0 / (max.x - min.x);
    let inv_range_y = 1.0 / (max.y - min.y);
    let inv_range_z = 1.0 / (max.z - min.z);
    
    for (point, morton) in points.iter().zip(output.iter_mut()) {
        // Normalize and scale in one step
        let x = ((point.x - min.x) * inv_range_x).clamp(0.0, 1.0) * scale;
        let y = ((point.y - min.y) * inv_range_y).clamp(0.0, 1.0) * scale;
        let z = ((point.z - min.z) * inv_range_z).clamp(0.0, 1.0) * scale;
        
        let x = prepare_morton_coordinate(x as u32);
        let y = prepare_morton_coordinate(y as u32);
        let z = prepare_morton_coordinate(z as u32);
        
        *morton = x | (y << 1) | (z << 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prepare_morton_coordinate() {
        // Test that prepare_morton_coordinate spreads bits correctly
        assert_eq!(prepare_morton_coordinate(0), 0);
        assert_eq!(prepare_morton_coordinate(1), 0x1);
        assert_eq!(prepare_morton_coordinate(2), 0x8);
        assert_eq!(prepare_morton_coordinate(3), 0x9);
        assert_eq!(prepare_morton_coordinate(1023), 0x09249249);
    }
    
    #[test]
    fn test_encode_morton() {
        let min = Point3::new(0.0, 0.0, 0.0);
        let max = Point3::new(1.0, 1.0, 1.0);
        
        // Test corner cases
        let origin = Point3::new(0.0, 0.0, 0.0);
        assert_eq!(encode_morton(&origin, &min, &max), 0);
        
        let corner = Point3::new(1.0, 1.0, 1.0);
        assert_eq!(encode_morton(&corner, &min, &max), 0x3FFFFFFF); // All 30 bits set
        
        // Test that nearby points have similar Morton codes
        let p1 = Point3::new(0.5, 0.5, 0.5);
        let p2 = Point3::new(0.51, 0.51, 0.51);
        let m1 = encode_morton(&p1, &min, &max);
        let m2 = encode_morton(&p2, &min, &max);
        
        // Different points should have different Morton codes
        assert_ne!(m1, m2, "Different points should have different Morton codes");
        
        // Test that identical points have identical codes
        let p3 = Point3::new(0.5, 0.5, 0.5);
        let m3 = encode_morton(&p3, &min, &max);
        assert_eq!(m1, m3, "Identical points should have identical Morton codes");
    }
    
    #[test]
    fn test_batch_encode_morton() {
        let min = Point3::new(0.0, 0.0, 0.0);
        let max = Point3::new(10.0, 10.0, 10.0);
        
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(10.0, 10.0, 10.0),
        ];
        
        let mut morton_codes = vec![0u32; points.len()];
        batch_encode_morton(&points, &min, &max, &mut morton_codes);
        
        // Verify batch encoding matches individual encoding
        for (i, point) in points.iter().enumerate() {
            assert_eq!(morton_codes[i], encode_morton(point, &min, &max));
        }
    }
} 