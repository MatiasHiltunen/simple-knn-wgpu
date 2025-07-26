// Morton Code Computation Shader
//
// This shader encodes 3D points into Morton codes (Z-order curve indices)
// for spatial sorting. Morton codes interleave the bits of normalized
// coordinates to preserve spatial locality.

struct Point3 {
    x: f32,
    y: f32,
    z: f32,
    _padding: f32,
}

struct BoundingBox {
    min: Point3,
    max: Point3,
}

// Input points
@group(0) @binding(0)
var<storage, read> points: array<Point3>;

// Global bounding box
@group(0) @binding(1)
var<uniform> global_bbox: BoundingBox;

// Output Morton codes
@group(0) @binding(2)
var<storage, read_write> morton_codes: array<u32>;

// Number of points
@group(0) @binding(3)
var<uniform> num_points: u32;

// Prepares a 10-bit coordinate for Morton encoding by spreading bits
fn prepare_morton_coord(x: u32) -> u32 {
    var v = x & 0x3FFu; // Ensure 10 bits
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v << 8u)) & 0x0300F00Fu;
    v = (v | (v << 4u)) & 0x030C30C3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

// Encodes a 3D point into a 30-bit Morton code
fn encode_morton(point: vec3<f32>, bbox_min: vec3<f32>, bbox_max: vec3<f32>) -> u32 {
    // Normalize coordinates to [0, 1]
    let range = bbox_max - bbox_min;
    let normalized = clamp((point - bbox_min) / range, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Scale to 10-bit integer range [0, 1023]
    let scale = 1023.0;
    let x = u32(normalized.x * scale);
    let y = u32(normalized.y * scale);
    let z = u32(normalized.z * scale);
    
    // Prepare each coordinate
    let xx = prepare_morton_coord(x);
    let yy = prepare_morton_coord(y);
    let zz = prepare_morton_coord(z);
    
    // Interleave bits: x=0,3,6,9...; y=1,4,7,10...; z=2,5,8,11...
    return xx | (yy << 1u) | (zz << 2u);
}

@compute @workgroup_size(256, 1, 1)
fn compute_morton(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx >= num_points) {
        return;
    }
    
    // Load point and bounding box
    let point = points[idx];
    let point_vec = vec3<f32>(point.x, point.y, point.z);
    let bbox_min = vec3<f32>(global_bbox.min.x, global_bbox.min.y, global_bbox.min.z);
    let bbox_max = vec3<f32>(global_bbox.max.x, global_bbox.max.y, global_bbox.max.z);
    
    // Compute and store Morton code
    morton_codes[idx] = encode_morton(point_vec, bbox_min, bbox_max);
} 