#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from the .cu file
extern "C" void axpby_launcher(const float* a,
                               const float* b,
                               float* out,
                               float alpha,
                               int n,
                               int config,
                               cudaStream_t stream);

// out = alpha * a + b
torch::Tensor axpby_forward(torch::Tensor a, torch::Tensor b, double alpha, int config) {
    // Normalize dtypes (the harness may regenerate inputs with its own seed
    // and dtype; never reject on dtype alone).
    if (a.dtype() != torch::kFloat32) a = a.to(torch::kFloat32);
    if (b.dtype() != torch::kFloat32) b = b.to(torch::kFloat32);

    // Normalize contiguity so the float4 view in the kernel is valid.
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();

    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto out = torch::empty_like(a);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    axpby_launcher(a.data_ptr<float>(),
                   b.data_ptr<float>(),
                   out.data_ptr<float>(),
                   static_cast<float>(alpha),
                   static_cast<int>(a.numel()),
                   config,
                   stream);

    return out;
}

void register_axpby(pybind11::module& m) {
    m.def("axpby_forward", &axpby_forward,
          "Fused axpby: out = alpha * a + b",
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha"),
          py::arg("config") = 0);
}

REGISTER_BINDING(axpby, register_axpby);
