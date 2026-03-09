// Fused SwiGLU Activation Compute Shader
//
// Replaces 3+ Burn dispatches (slice gate, slice value, silu, mul) with a single
// GPU dispatch per SwiGLU call.
// Called 32 times per temporal step + 6*16=96 times per depth step = 128 per frame.
//
// Algorithm:
//   Given input of size 2*N (concatenated gate and value from linear_in):
//   output[i] = silu(input[i]) * input[i + N]
//   where silu(x) = x / (1.0 + exp(-x))
//
// Bindings:
//   @binding(0) input:  f32 array [2 * N]
//   @binding(1) output: f32 array [N]
//   @binding(2) info:   u32 array [1] = { N }
//
// Dispatch: workgroups = ceil(N / 256), workgroup_size = (256, 1, 1)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> info: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = info[0];
    let idx = gid.x;
    if (idx >= N) {
        return;
    }

    let gate_val = input[idx];
    let value_val = input[idx + N];

    // silu(x) = x / (1.0 + exp(-x)) = x * sigmoid(x)
    let sigmoid_gate = 1.0 / (1.0 + exp(-gate_val));
    let silu_gate = gate_val * sigmoid_gate;

    output[idx] = silu_gate * value_val;
}
