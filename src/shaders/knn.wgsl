// K-Nearest Neighbors Search Shader
//
// This shader performs the actual KNN search for each point. It uses the
// spatial partitioning from Morton sorting and box bounding boxes to
// efficiently find the 3 nearest neighbors.

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

// Sorted indices
@group(0) @binding(1)
var<storage, read> sorted_indices: array<u32>;

// Box bounding boxes
@group(0) @binding(2)
var<storage, read> box_bboxes: array<BoundingBox>;

// Output distances
@group(0) @binding(3)
var<storage, read_write> distances: array<f32>;

// Configuration: x=num_points, y=box_size, z=num_boxes, w=k (always 3)
@group(0) @binding(4)
var<uniform> config: vec4<u32>;

// Computes squared distance between two points
fn distance_squared(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let diff = a - b;
    return dot(diff, diff);
}

// Computes squared distance from a box to a point
fn box_distance_squared(box: BoundingBox, point: vec3<f32>) -> f32 {
    let box_min = vec3<f32>(box.min.x, box.min.y, box.min.z);
    let box_max = vec3<f32>(box.max.x, box.max.y, box.max.z);
    
    let dx = max(0.0, max(box_min.x - point.x, point.x - box_max.x));
    let dy = max(0.0, max(box_min.y - point.y, point.y - box_max.y));
    let dz = max(0.0, max(box_min.z - point.z, point.z - box_max.z));
    
    return dx * dx + dy * dy + dz * dz;
}

// Updates the k‐best distances array with a new candidate, skipping duplicates.
fn update_k_best(
    best_dists: ptr<function, array<f32, 3>>,
    best_indices: ptr<function, array<u32, 3>>,
    new_dist: f32,
    new_idx: u32,
) {
    // Skip if we've already stored this index (prevents duplicates).
    if (new_idx == (*best_indices)[0] || new_idx == (*best_indices)[1] || new_idx == (*best_indices)[2]) {
        return;
    }

    if (new_dist < (*best_dists)[0]) {
        (*best_dists)[2]   = (*best_dists)[1];
        (*best_indices)[2] = (*best_indices)[1];

        (*best_dists)[1]   = (*best_dists)[0];
        (*best_indices)[1] = (*best_indices)[0];

        (*best_dists)[0]   = new_dist;
        (*best_indices)[0] = new_idx;
    } else if (new_dist < (*best_dists)[1]) {
        (*best_dists)[2]   = (*best_dists)[1];
        (*best_indices)[2] = (*best_indices)[1];

        (*best_dists)[1]   = new_dist;
        (*best_indices)[1] = new_idx;
    } else if (new_dist < (*best_dists)[2]) {
        (*best_dists)[2]   = new_dist;
        (*best_indices)[2] = new_idx;
    }
}

@compute @workgroup_size(256, 1, 1)
fn compute_knn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let num_points = config.x;
    let box_size = config.y;
    let num_boxes = config.z;
    
    if (idx >= num_points) {
        return;
    }
    
    // Get the query point
    let query_idx = sorted_indices[idx];
    let query_point = points[query_idx];
    let query_vec = vec3<f32>(query_point.x, query_point.y, query_point.z);
    
    var best_dists: array<f32, 3>;
    var best_indices: array<u32, 3>;

    // Initialize distances to large values and indices to an invalid sentinel.
    best_dists[0] = 1e10;
    best_dists[1] = 1e10;
    best_dists[2] = 1e10;

    best_indices[0] = 0xffffffffu;
    best_indices[1] = 0xffffffffu;
    best_indices[2] = 0xffffffffu;
    
    // First, check immediate neighbors in the sorted array
    // This is very efficient due to spatial locality from Morton ordering
    let neighbor_range = 3u;
    let start_idx = select(0u, idx - neighbor_range, idx > neighbor_range);
    let end_idx = min(idx + neighbor_range + 1u, num_points);
    
    for (var i = start_idx; i < end_idx; i++) {
        if (i == idx) {
            continue;
        }
        
        let neighbor_idx = sorted_indices[i];
        let neighbor = points[neighbor_idx];
        let neighbor_vec = vec3<f32>(neighbor.x, neighbor.y, neighbor.z);
        let dist_sq = distance_squared(query_vec, neighbor_vec);
        
        if (dist_sq > 0.0) { // Avoid self-distance
            update_k_best(&best_dists, &best_indices, dist_sq, neighbor_idx);
        }
    }
    
    // Use the worst current distance as rejection threshold
    let reject_dist = best_dists[2];
    
    // Now search all boxes that might contain closer points
    for (var box_id = 0u; box_id < num_boxes; box_id++) {
        let box = box_bboxes[box_id];
        let box_dist = box_distance_squared(box, query_vec);
        
        // Skip boxes that are too far away
        if (box_dist > reject_dist || box_dist > best_dists[2]) {
            continue;
        }
        
        // Search all points in this box
        let box_start = box_id * box_size;
        let box_end = min(box_start + box_size, num_points);
        
        for (var i = box_start; i < box_end; i++) {
            let point_idx = sorted_indices[i];
            if (point_idx == query_idx) {
                continue;
            }
            
            let point = points[point_idx];
            let point_vec = vec3<f32>(point.x, point.y, point.z);
            let dist_sq = distance_squared(query_vec, point_vec);
            
            if (dist_sq > 0.0) {
                update_k_best(&best_dists, &best_indices, dist_sq, point_idx);
            }
        }
    }
    
    // Compute arithmetic mean of actual distances (not squared)
    let mean_dist = (sqrt(best_dists[0]) + sqrt(best_dists[1]) + sqrt(best_dists[2])) / 3.0;
    distances[query_idx] = mean_dist;
} 