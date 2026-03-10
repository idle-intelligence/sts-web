// Fused SwiGLU Activation Compute Shader
//
// Replaces multiple Burn dispatches (slice + sigmoid + mul + mul) with a single
// GPU dispatch. Used in every FFN block of both temporal and depth transformers.
//
// Input:  combined[half_dim * 2] — f32 array, gate[0..half_dim] || value[half_dim..2*half_dim]
// Output: result[half_dim] — f32 array
// Info:   [half_dim: u32]
//
// Formula: output[i] = silu(gate[i]) * value[i]
//          where silu(x) = x / (1.0 + exp(-x))
//
// Dispatch: workgroups = (ceil(half_dim / 256), 1, 1), workgroup_size = (256, 1, 1)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> info: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_dim = info[0];
    let i = gid.x;

    if (i >= half_dim) {
        return;
    }

    let gate = input[i];
    let value = input[half_dim + i];

    // silu(x) = x * sigmoid(x) = x / (1.0 + exp(-x))
    let silu_gate = gate / (1.0 + exp(-gate));

    output[i] = silu_gate * value;
}
