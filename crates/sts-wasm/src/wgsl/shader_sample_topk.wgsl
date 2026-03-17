// GPU Top-K Sampling Compute Shader
//
// Performs temperature-scaled top-k sampling with repetition penalty entirely
// on GPU, returning a single u32 token index. Avoids reading back full logits.
//
// Algorithm:
// 1. Apply repetition penalty to specified token IDs
// 2. Apply temperature scaling
// 3. Find top-k values via parallel partial sort
// 4. Compute softmax over top-k candidates
// 5. Sample from distribution using provided random number
//
// Input:
//   logits[V]        — f32 logits (vocab size)
//   penalty_tokens[] — u32 token IDs to penalize, terminated by 0xFFFFFFFF
//   info[0]          — V (vocab size)
//   info[1]          — K (top-k, max 256)
//   info[2]          — temperature as f32 bits
//   info[3]          — penalty as f32 bits
//   info[4]          — random value [0,1) as f32 bits
//   info[5]          — num_penalty_tokens
//
// Output:
//   result[0]        — sampled token index (u32, stored as u32 bits)
//
// Dispatch: 1 workgroup, workgroup_size(256)

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> penalty_tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
@group(0) @binding(3) var<storage, read> info: array<u32>;

var<workgroup> shared_vals: array<f32, 256>;
var<workgroup> shared_idxs: array<u32, 256>;
// For top-k collection: store the k-th threshold
var<workgroup> topk_threshold: f32;
// For softmax reduction
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let V = info[0];
    let K = info[1];
    let temperature = bitcast<f32>(info[2]);
    let penalty = bitcast<f32>(info[3]);
    let rand_val = bitcast<f32>(info[4]);
    let num_penalty = info[5];

    // Step 1: Apply repetition penalty (each thread handles a chunk)
    let penalty_per_thread = (num_penalty + 255u) / 256u;
    let penalty_start = tid * penalty_per_thread;
    let penalty_end = min(penalty_start + penalty_per_thread, num_penalty);
    for (var p = penalty_start; p < penalty_end; p = p + 1u) {
        let tok = penalty_tokens[p];
        if (tok < V) {
            let val = logits[tok];
            if (val > 0.0) {
                logits[tok] = val / penalty;
            } else {
                logits[tok] = val * penalty;
            }
        }
    }
    workgroupBarrier();

    // Step 2: Apply temperature and find local max
    let stride = 256u;
    var local_max: f32 = -3.402823466e+38;
    var local_idx: u32 = 0u;

    var i = tid;
    while (i < V) {
        logits[i] = logits[i] / temperature;
        let val = logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
        i = i + stride;
    }

    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    workgroupBarrier();

    // Step 3: Find the global maximum (tree reduction)
    var half = 128u;
    while (half > 0u) {
        if (tid < half) {
            if (shared_vals[tid + half] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + half];
                shared_idxs[tid] = shared_idxs[tid + half];
            }
        }
        workgroupBarrier();
        half = half >> 1u;
    }

    // Now shared_vals[0] = global max, shared_idxs[0] = argmax

    // For K=1 or simple case, just return argmax
    // (handles edge case and is the greedy path)
    if (K <= 1u) {
        if (tid == 0u) {
            result[0] = shared_idxs[0];
        }
        return;
    }

    // Step 4: Find top-K threshold using iterative approach
    // We'll find the K-th largest value by counting elements >= threshold.
    // Start with the global max and binary search down.
    //
    // Alternative simpler approach: each thread collects top-K from its stripe,
    // then merge. But for K up to 256 and V=2048, the simplest correct approach
    // is: each thread scans all V elements and we pick the K-th largest.
    //
    // Since V=2048 is small, thread 0 can just do a simple partial sort.
    // This is faster than a complex parallel algorithm for small V.

    if (tid == 0u) {
        // Thread 0: find top-K indices and sample
        // For V=2048, K=50 (typical), this is ~2048 comparisons + sort of 50

        // Use shared memory as scratch for top-K candidates
        // We'll store up to K (idx, val) pairs
        // Since K <= 256, fits in shared memory

        // Initialize with -inf
        for (var j = 0u; j < K; j = j + 1u) {
            shared_vals[j] = -3.402823466e+38;
            shared_idxs[j] = 0u;
        }

        // Track minimum of current top-K
        var min_topk: f32 = -3.402823466e+38;
        var min_topk_pos: u32 = 0u;

        // First K elements go directly in
        let first_k = min(K, V);
        for (var j = 0u; j < first_k; j = j + 1u) {
            shared_vals[j] = logits[j];
            shared_idxs[j] = j;
        }

        // Find initial min
        min_topk = shared_vals[0];
        min_topk_pos = 0u;
        for (var j = 1u; j < first_k; j = j + 1u) {
            if (shared_vals[j] < min_topk) {
                min_topk = shared_vals[j];
                min_topk_pos = j;
            }
        }

        // Scan remaining elements
        for (var j = K; j < V; j = j + 1u) {
            let val = logits[j];
            if (val > min_topk) {
                // Replace the minimum
                shared_vals[min_topk_pos] = val;
                shared_idxs[min_topk_pos] = j;
                // Find new minimum
                min_topk = shared_vals[0];
                min_topk_pos = 0u;
                for (var m = 1u; m < K; m = m + 1u) {
                    if (shared_vals[m] < min_topk) {
                        min_topk = shared_vals[m];
                        min_topk_pos = m;
                    }
                }
            }
        }

        // Now shared_vals/shared_idxs[0..K] contain top-K logits

        // Sort top-K by descending value (insertion sort, K is small)
        // Required so cumulative probability sampling matches CPU path
        for (var si = 1u; si < K; si = si + 1u) {
            let key_val = shared_vals[si];
            let key_idx = shared_idxs[si];
            var sj: i32 = i32(si) - 1;
            while (sj >= 0 && shared_vals[u32(sj)] < key_val) {
                shared_vals[u32(sj + 1)] = shared_vals[u32(sj)];
                shared_idxs[u32(sj + 1)] = shared_idxs[u32(sj)];
                sj = sj - 1;
            }
            shared_vals[u32(sj + 1)] = key_val;
            shared_idxs[u32(sj + 1)] = key_idx;
        }

        // Softmax over top-K
        // Find max for numerical stability
        var max_val: f32 = shared_vals[0];
        for (var j = 1u; j < K; j = j + 1u) {
            if (shared_vals[j] > max_val) {
                max_val = shared_vals[j];
            }
        }

        var exp_sum: f32 = 0.0;
        for (var j = 0u; j < K; j = j + 1u) {
            let e = exp(shared_vals[j] - max_val);
            shared_vals[j] = e;  // reuse for probabilities
            exp_sum = exp_sum + e;
        }

        // Sample from distribution
        var cumulative: f32 = 0.0;
        var sampled_token: u32 = shared_idxs[K - 1u]; // default to last
        for (var j = 0u; j < K; j = j + 1u) {
            cumulative = cumulative + shared_vals[j] / exp_sum;
            if (rand_val < cumulative) {
                sampled_token = shared_idxs[j];
                break;
            }
        }

        result[0] = sampled_token;
    }
}
