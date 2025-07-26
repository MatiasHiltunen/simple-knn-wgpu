// Box Bounding Box Computation Shader
//
// This shader computes axis-aligned bounding boxes for each spatial partition
// after points have been sorted by Morton code. Each workgroup processes one
// box of points and computes its tight bounding box.

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

// Sorted points (by Morton code)
@group(0) @binding(0)
var<storage, read> points: array<Point3>;

// Sorted indices (mapping to original point order)
@group(0) @binding(1)
var<storage, read> sorted_indices: array<u32>;

// Output box bounding boxes
@group(0) @binding(2)
var<storage, read_write> box_bboxes: array<BoundingBox>;

// Configuration
@group(0) @binding(3)
var<uniform> config: vec4<u32>; // x: num_points, y: box_size, z: num_boxes

// Shared memory for reduction
var<workgroup> shared_min: array<vec3<f32>, 1024>;
var<workgroup> shared_max: array<vec3<f32>, 1024>;

@compute @workgroup_size(256, 1, 1)
fn compute_box_bbox(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let box_id = workgroup_id.x;
    let num_points = config.x;
    let box_size = config.y;
    let num_boxes = config.z;
    
    if (box_id >= num_boxes) {
        return;
    }
    
    // Calculate the range of points for this box
    let box_start = box_id * box_size;
    let box_end = min(box_start + box_size, num_points);
    let box_points = box_end - box_start;
    
    // Initialize with extreme values
    var local_min = vec3<f32>(1e10, 1e10, 1e10);
    var local_max = vec3<f32>(-1e10, -1e10, -1e10);
    
    // Each thread processes multiple points if necessary
    let points_per_thread = (box_points + 255u) / 256u;
    let thread_start = box_start + tid * points_per_thread;
    let thread_end = min(thread_start + points_per_thread, box_end);
    
    // Process assigned points
    for (var i = thread_start; i < thread_end; i++) {
        let idx = sorted_indices[i];
        let p = points[idx];
        let point_vec = vec3<f32>(p.x, p.y, p.z);
        local_min = min(local_min, point_vec);
        local_max = max(local_max, point_vec);
    }
    
    // Store in shared memory
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }
    
    // Thread 0 writes the result
    if (tid == 0u) {
        box_bboxes[box_id].min.x = shared_min[0].x;
        box_bboxes[box_id].min.y = shared_min[0].y;
        box_bboxes[box_id].min.z = shared_min[0].z;
        box_bboxes[box_id].min._padding = 0.0;
        
        box_bboxes[box_id].max.x = shared_max[0].x;
        box_bboxes[box_id].max.y = shared_max[0].y;
        box_bboxes[box_id].max.z = shared_max[0].z;
        box_bboxes[box_id].max._padding = 0.0;
    }
} 