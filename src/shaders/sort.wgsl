// Radix Sort Shader
//
// Implements a simple GPU radix sort for 32-bit keys.
// The algorithm performs one pass per bit using atomic counters
// and two compute stages (count and reorder).

struct Params {
    num: u32;
    bit: u32;
};

@group(0) @binding(0)
var<storage, read> in_keys: array<u32>;
@group(0) @binding(1)
var<storage, read> in_indices: array<u32>;
@group(0) @binding(2)
var<storage, read_write> prefix: array<u32>;
@group(0) @binding(3)
var<storage, read_write> counts: array<atomic<u32>, 2>;
@group(0) @binding(4)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn radix_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num) { return; }

    let key = in_keys[idx];
    let bit = (key >> params.bit) & 1u;

    if (bit == 0u) {
        prefix[idx] = atomicAdd(&counts[0], 1u);
    } else {
        prefix[idx] = atomicAdd(&counts[1], 1u);
    }
}

@group(0) @binding(0)
var<storage, read> r_in_keys: array<u32>;
@group(0) @binding(1)
var<storage, read> r_in_indices: array<u32>;
@group(0) @binding(2)
var<storage, read> r_prefix: array<u32>;
@group(0) @binding(3)
var<storage, read> r_counts: array<atomic<u32>, 2>;
@group(0) @binding(4)
var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(5)
var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(6)
var<uniform> r_params: Params;

@compute @workgroup_size(256)
fn radix_reorder(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= r_params.num) { return; }

    let key = r_in_keys[idx];
    let bit = (key >> r_params.bit) & 1u;
    let p = r_prefix[idx];
    let zeros = atomicLoad(&r_counts[0]);

    var dst = p;
    if (bit == 1u) {
        dst = zeros + p;
    }

    out_keys[dst] = key;
    out_indices[dst] = r_in_indices[idx];
}
