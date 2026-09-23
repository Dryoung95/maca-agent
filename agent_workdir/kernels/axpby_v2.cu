
#include <cuda_runtime.h>

__global__ void axpby_kernel(
    float* __restrict__ out, const float* __restrict__ a,
    const float* __restrict__ b, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grid = blockDim.x * gridDim.x;
    for (int j = i; j < n; j += grid) {
        out[j] = alpha * a[j] + b[j];
    }
}

extern "C" void axpby_v2_launcher(
    float* out, const float* a, const float* b, float alpha,
    int n, cudaStream_t stream)
{
    if (n <= 0) return;
    axpby_kernel<<<1024, 256, 0, stream>>>(out, a, b, alpha, n);
}
