// Bounding Box Computation Shader
// 
// This shader computes the global bounding box of all input points using
// a parallel reduction algorithm. Each workgroup processes a chunk of points
// and produces a local bounding box, which are then combined.

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

// Input points buffer
@group(0) @binding(0)
var<storage, read> points: array<Point3>;

// Output bounding boxes (one per workgroup)
@group(0) @binding(1)
var<storage, read_write> workgroup_boxes: array<BoundingBox>;

// Uniform buffer containing the number of points
@group(0) @binding(2)
var<uniform> num_points: u32;

// Shared memory for workgroup reduction
var<workgroup> shared_min: array<vec3<f32>, 256>;
var<workgroup> shared_max: array<vec3<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn compute_bbox(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let wg_id = workgroup_id.x;
    
    // Initialize with extreme values
    var local_min = vec3<f32>(1e10, 1e10, 1e10);
    var local_max = vec3<f32>(-1e10, -1e10, -1e10);
    
    // Each thread processes multiple points to improve efficiency
    let points_per_thread = (num_points + 256u * num_workgroups.x - 1u) / (256u * num_workgroups.x);
    let start_idx = gid * points_per_thread;
    let end_idx = min(start_idx + points_per_thread, num_points);
    
    // Process assigned points
    for (var i = start_idx; i < end_idx; i++) {
        let p = points[i];
        let point_vec = vec3<f32>(p.x, p.y, p.z);
        local_min = min(local_min, point_vec);
        local_max = max(local_max, point_vec);
    }
    
    // Store in shared memory
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    workgroupBarrier();
    
    // Parallel reduction within workgroup
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }
    
    // Thread 0 writes the workgroup result
    if (tid == 0u) {
        workgroup_boxes[wg_id].min.x = shared_min[0].x;
        workgroup_boxes[wg_id].min.y = shared_min[0].y;
        workgroup_boxes[wg_id].min.z = shared_min[0].z;
        workgroup_boxes[wg_id].min._padding = 0.0;
        
        workgroup_boxes[wg_id].max.x = shared_max[0].x;
        workgroup_boxes[wg_id].max.y = shared_max[0].y;
        workgroup_boxes[wg_id].max.z = shared_max[0].z;
        workgroup_boxes[wg_id].max._padding = 0.0;
    }
}

// Second pass: combine workgroup results
@compute @workgroup_size(1, 1, 1)
fn reduce_bbox(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    if (global_id.x != 0u) {
        return;
    }
    
    var global_min = vec3<f32>(1e10, 1e10, 1e10);
    var global_max = vec3<f32>(-1e10, -1e10, -1e10);
    
    // Combine all workgroup boxes
    let num_workgroups = arrayLength(&workgroup_boxes);
    for (var i = 0u; i < num_workgroups; i++) {
        let box = workgroup_boxes[i];
        global_min = min(global_min, vec3<f32>(box.min.x, box.min.y, box.min.z));
        global_max = max(global_max, vec3<f32>(box.max.x, box.max.y, box.max.z));
    }
    
    // Write final result to first element
    workgroup_boxes[0].min.x = global_min.x;
    workgroup_boxes[0].min.y = global_min.y;
    workgroup_boxes[0].min.z = global_min.z;
    workgroup_boxes[0].min._padding = 0.0;
    
    workgroup_boxes[0].max.x = global_max.x;
    workgroup_boxes[0].max.y = global_max.y;
    workgroup_boxes[0].max.z = global_max.z;
    workgroup_boxes[0].max._padding = 0.0;
} 