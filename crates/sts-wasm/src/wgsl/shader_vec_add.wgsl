// Element-wise Vector Addition Compute Shader
//
// Used for residual connections: output[i] = a[i] + b[i].
// Replaces Burn's add dispatch with a single custom dispatch.
//
// Input A: f32 array [len]
// Input B: f32 array [len]
// Output:  f32 array [len] (can alias A or B for in-place)
// Info:    [len: u32]
//
// Dispatch: workgroups = (ceil(len / 256), 1, 1), workgroup_size = (256, 1, 1)

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = info[0];
    let i = gid.x;

    if (i >= len) {
        return;
    }

    output[i] = a[i] + b[i];
}
