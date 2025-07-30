// Radix Sort Shader
//
// Implements a simple GPU radix sort for 32-bit keys.
// This version processes 4 bits per pass (radix 16) to reduce
// the number of required compute dispatches compared to the
// original bit-by-bit implementation. Each pass consists of
// three stages: counting, prefix sum, and reorder.

struct Params {
    num: u32,
    shift: u32,
};

@group(0) @binding(0)
var<storage, read> in_keys: array<u32>;
@group(0) @binding(1)
var<storage, read_write> prefix: array<u32>;
@group(0) @binding(2)
// One counter per radix bucket (16 buckets for 4 bits)
var<storage, read_write> counts: array<atomic<u32>, 16>;
@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn radix_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num) { return; }

    let key = in_keys[idx];
    let bin = (key >> params.shift) & 0xFu;
    prefix[idx] = atomicAdd(&counts[bin], 1u);
}

@group(0) @binding(0)
// Atomic counts from the first pass
var<storage, read_write> p_counts: array<atomic<u32>, 16>;
@group(0) @binding(1)
// Output prefix offsets per bucket
var<storage, read_write> p_offsets: array<u32>;

// Computes exclusive prefix sums of the counts buffer.
@compute @workgroup_size(1)
fn radix_prefix() {
    var sum: u32 = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let c = atomicLoad(&p_counts[i]);
        p_offsets[i] = sum;
        sum = sum + c;
    }
}

@group(0) @binding(0)
var<storage, read> r_in_keys: array<u32>;
@group(0) @binding(1)
var<storage, read> r_in_indices: array<u32>;
@group(0) @binding(2)
var<storage, read> r_prefix: array<u32>;
@group(0) @binding(3)
// Atomic variables require read_write access even if only read
var<storage, read> r_offsets: array<u32>;
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
    let bin = (key >> r_params.shift) & 0xFu;
    let p = r_prefix[idx];
    let dst = r_offsets[bin] + p;

    out_keys[dst] = key;
    out_indices[dst] = r_in_indices[idx];
}
