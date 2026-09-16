#!/usr/bin/env python3
"""Smoke test for agent_loop.py.

Feeds the harness a scripted known-good attempt on the axpby example and
checks that every stage reports what it should: compile passes, correctness
5/5, profiling produces sane numbers, and the KernelBench gate fires.

This validates the harness, not the kernel — the attempt is written by hand
and known to work. If the harness reports a failure here, the loop itself is
broken and any later model rollout would be mismeasured.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_loop import Attempt, run_episode, compute_reward, AttemptResult

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

KERNEL = r'''
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
'''

BINDING = r'''
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
'''

MODEL_NEW = '''import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return cuda_extension.axpby_v2_forward(a, b, self.alpha, 0)
'''


def main():
    good = Attempt(
        name="scalar_axpby_v2",
        kernel_files={"kernels/axpby_v2.cu": KERNEL},
        binding_files={"kernels/axpby_v2_binding.cpp": BINDING},
        model_new=MODEL_NEW,
    )

    print(f"workspace: {WORKSPACE}")
    summary = run_episode(WORKSPACE, [good], task_name="smoke_axpby")

    a = summary["attempts"][0]
    checks = []
    checks.append(("compiled", a["compiled"] is True))
    checks.append(("correct 5/5", a["correct"] is True))
    checks.append(("ext measured", a["ext_us"] > 0))
    checks.append(("compile baseline measured", a["compile_us"] > 0))
    checks.append(("reward > 0", a["reward"] > 0))
    checks.append(("no error", not a["error"]))

    print("\n=== smoke test results ===")
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed
    print(f"\noverall: {'SMOKE PASS' if ok else 'SMOKE FAIL'}")
    print(json.dumps(summary, indent=2, default=str))

    with open(os.path.join(WORKSPACE, "smoke_report.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
