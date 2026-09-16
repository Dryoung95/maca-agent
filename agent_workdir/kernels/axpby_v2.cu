#include <cuda_runtime.h>

// B1: vectorized axpby — float4 loads/stores, grid-stride, warp-64 friendly blocks.
// On C500 a warp is 64 threads, so 256 threads = 4 warps; vectorization widens the
// per-thread work to 16 floats so the kernel is compute/BW bound rather than
// launch-overhead bound at the sizes the RL reward can actually distinguish.

__device__ __forceinline__ float4 operator*(float s, float4 v) {
    return make_float4(s * v.x, s * v.y, s * v.z, s * v.w);
}

__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template<int THREADS>
__global__ void axpby_kernel_v2(float* __restrict__ out,
                                const float* __restrict__ a,
                                const float* __restrict__ b,
                                float alpha, int size) {
    // scalar tail handled by a separate simple pass; this kernel sees only
    // the float4-aligned prefix
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = size >> 2;  // number of float4 elements
    int stride = blockDim.x * gridDim.x;
    float4* out4 = reinterpret_cast<float4*>(out);
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);

    for (int i = tid; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vb = b4[i];
        out4[i] = alpha * va + vb;
    }
    // scalar remainder
    int rem_start = n4 << 2;
    int tid_s = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = rem_start + tid_s; j < size; j += stride) {
        out[j] = alpha * a[j] + b[j];
    }
}

extern "C" void axpby_v2_launcher(
    float* out,
    const float* a,
    const float* b,
    float alpha,
    int size,
    int config,
    cudaStream_t stream
) {
    if (size <= 0) return;
    int n4 = size >> 2;
    int aligned = n4 << 2;
    int scalar_tail = size - aligned;
    switch (config) {
        case 1: {
            int threads = 128;
            int blocks = (n4 + threads - 1) / threads;
            axpby_kernel_v2<128><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        case 2: {
            int threads = 512;
            int blocks = (n4 + threads - 1) / threads;
            axpby_kernel_v2<512><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        default: {
            int threads = 256;
            int blocks = (n4 + threads - 1) / threads;
            axpby_kernel_v2<256><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
    }
}
