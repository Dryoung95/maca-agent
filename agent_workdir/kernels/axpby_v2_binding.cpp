
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void axpby_v2_launcher(
    float* out, const float* a, const float* b, float alpha,
    int n, cudaStream_t stream);

static torch::Tensor axpby_v2_forward(torch::Tensor a, torch::Tensor b,
                                      double alpha, int64_t unused) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    if (a.dtype() != torch::kFloat32) a = a.to(torch::kFloat32);
    if (b.dtype() != torch::kFloat32) b = b.to(torch::kFloat32);
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();
    auto out = torch::empty_like(a);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    axpby_v2_launcher(out.data_ptr<float>(), a.data_ptr<float>(),
                      b.data_ptr<float>(), (float)alpha,
                      (int)a.numel(), stream);
    return out;
}

static void register_axpby_v2(pybind11::module& m) {
    m.def("axpby_v2_forward", &axpby_v2_forward,
          py::arg("a"), py::arg("b"), py::arg("alpha"), py::arg("unused"));
}

REGISTER_BINDING(axpby_v2, register_axpby_v2);
