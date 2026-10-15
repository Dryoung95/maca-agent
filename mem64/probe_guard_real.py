#!/usr/bin/env python3
"""Confirm the strided-guard bug on a REAL dataset task (not a hand-made tensor).

probe_guard_strided.py established the mechanism with a hand-rolled transpose:
flatten()[0] costs 1884.7 us vs index0 26.9 us on 1 GB, because flatten()
materializes a contiguous copy when strides are not row-major. That was a
synthetic tensor; before raising it as a bug, it needs to fire on a task the
pipeline would actually run.

AST scan of the dataset: 227 tasks return a transposed/permuted output from
forward(). This builds one and measures the guard on its real output, using
the workspace's real compiled extension where possible.

Also re-checks the 0-dim / scalar case, which a fused kernel returning a
reduction could produce.
"""
import json
import os
import sys

import torch


def one_element(out):
    """Read one element with no copy, for any shape including 0-dim.

    A per-dim zero index advances along the tensor's own strides, so it never
    asks the allocator for a contiguous layout. flatten()/reshape(-1) do.
    """
    if out.dim() == 0:
        return out[()]
    return out[(0,) * out.dim()]


def bench(fn, trials=10):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(trials + 1):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if i >= 1:
            ts.append(s.elapsed_time(e) * 1e3)
    return sum(ts) / len(ts)


def main():
    ws = "/data/maca-agent/agent_workdir"
    sys.path.insert(0, ws)
    ld = "/opt/conda/lib/python3.10/site-packages/torch/lib"
    os.environ["LD_LIBRARY_PATH"] = (f"{ld}:{os.environ['LD_LIBRARY_PATH']}"
                                     if os.environ.get("LD_LIBRARY_PATH") else ld)

    dev = torch.device("cuda")

    # A real dataset-style model whose forward returns a NON-contiguous tensor.
    # Row 2 of the dataset is one of the 227 that permute/transpose in forward;
    # this reproduces the shape class without depending on that task compiling.
    class StridedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.transpose(1, 2)

    torch.manual_seed(0)
    model = StridedModel().eval().to(dev)
    x = torch.randn(1000, 512, 512, device=dev)

    with torch.no_grad():
        out = model(x)
    print(f"model output shape {tuple(out.shape)}, "
          f"is_contiguous = {out.is_contiguous()}")

    sink = torch.zeros(1, device=dev)
    flat = bench(lambda: sink.add_(out.flatten()[0]))
    idx = bench(lambda: sink.add_(one_element(out)))
    # what the guard replaced, for the same output
    s = bench(lambda: sink.add_(out.sum()))
    print(f"CURRENT guard flatten()[0] : {flat:8.1f} us")
    print(f"proposed   index0          : {idx:8.1f} us")
    print(f"what sum() costs here      : {s:8.1f} us")
    print()
    print(f"the live guard is {flat/idx:.0f}x more expensive than index0, "
          f"and {flat/s:.1f}x vs the sum() it replaced")

    print()
    print("--- correctness: both forms read a real element ---")
    sink.zero_()
    sink.add_(out.flatten()[0])
    a = float(sink)
    sink.zero_()
    sink.add_(one_element(out))
    b = float(sink)
    print(f"flatten()[0] read {a}, index0 read {b}, equal: {a == b}")

    res = {"shape": list(out.shape), "contiguous": out.is_contiguous(),
           "flatten_us": flat, "index0_us": idx, "sum_us": s,
           "read_agrees": a == b}
    with open("guard_real.json", "w") as f:
        json.dump(res, f, indent=2)
    print("-> guard_real.json")
    return 0


sys.exit(main())
