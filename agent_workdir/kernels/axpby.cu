// Fused axpby: out = alpha * a + b
// Memory-bound elementwise op: single pass, float4 vectorized loads/stores,
// grid-stride loop so a bounded grid covers arbitrarily large tensors.
#include <cuda_runtime.h>

template <int BLOCK>
__global__ void axpby_kernel(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ out,
                             float alpha,
                             int n) {
    const int n4    = n >> 2;           // number of complete float4 groups
    const int tail  = n - (n4 << 2);    // 0..3 leftover elements
    const int tid   = blockIdx.x * BLOCK + threadIdx.x;
    const int stride = gridDim.x * BLOCK;

    // Vectorised body: 4 floats processed per thread per iteration.
    // Index 4*i sits at byte offset 16*i, so these loads/stores are
    // 16-byte aligned and fully coalesced.
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4*       o4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < n4; i += stride) {
        const float4 va = a4[i];
        const float4 vb = b4[i];
        float4 r;
        r.x = fmaf(alpha, va.x, vb.x);
        r.y = fmaf(alpha, va.y, vb.y);
        r.z = fmaf(alpha, va.z, vb.z);
        r.w = fmaf(alpha, va.w, vb.w);
        o4[i] = r;
    }

    // Scalar tail for the final <4 elements (tail < stride, so each
    // element is touched exactly once).
    const int tail_base = n4 << 2;
    for (int j = tid; j < tail; j += stride) {
        const int idx = tail_base + j;
        out[idx] = fmaf(alpha, a[idx], b[idx]);
    }
}

// C-interface launcher (no PyTorch dependencies).
// config selects the block size so it can be tuned without recompiling.
extern "C" void axpby_launcher(const float* a,
                               const float* b,
                               float* out,
                               float alpha,
                               int n,
                               int config,
                               cudaStream_t stream) {
    const int MAX_BLOCKS = 1024; // bounded grid; grid-stride covers the rest

    int groups = (n + 3) >> 2;   // ceil(n/4) float4 groups
    int blocks = groups > 0 ? (groups + 255) / 256 : 1;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    if (blocks < 1) blocks = 1;

    switch (config) {
        case 1:
            axpby_kernel<128><<<blocks, 128, 0, stream>>>(a, b, out, alpha, n);
            break;
        case 2:
            axpby_kernel<512><<<blocks, 512, 0, stream>>>(a, b, out, alpha, n);
            break;
        case 0:
        default:
            axpby_kernel<256><<<blocks, 256, 0, stream>>>(a, b, out, alpha, n);
            break;
    }
}
