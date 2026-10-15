#!/usr/bin/env python3
"""The 1-element guard is NOT safe on non-contiguous outputs.

utils/profiling.py now does `out.flatten()[0]`. That is ~25 us on a contiguous
output but `.flatten()` materializes a CONTIGUOUS COPY when strides are not
already row-major, so on a strided output it costs a full read+write pass.
Measured on a 1 GB tensor:

  contiguous output: flatten()[0]   25.0 us, sum()  696.6 us  -> guard works
  strided output:    flatten()[0] 1892.8 us, sum()  698.3 us  -> 2.7x WORSE
                                                           than the bug it fixed

The row311 kernel writes contiguous output, which is exactly why the fix looked
free there. It does not generalize: fused/optimized kernels in this dataset do
return transposed or strided tensors, and those are the outputs most likely to
be non-contiguous.

What we actually need is a read of ONE element at the tensor's own strides.
Indexing with a per-dim zero does that -- no copy, because it never asks for
a contiguous layout. This verifies the replacement against all the shapes a
model can return: contiguous, transposed, sliced, 1-D, and 0-dim (a scalar
output, where indexing with `(0,)` must NOT be used).
"""
import json
import os
import sys

import torch


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


def one_element(out):
    """Read one element with no copy, for any shape including 0-dim.

    A per-dim zero index advances along the tensor's own strides, so it never
    asks the allocator for a contiguous layout. flatten()/reshape(-1) both do.
    """
    if out.dim() == 0:
        # a 0-dim tensor has exactly one element; indexing it with () reads it
        return out[()]
    return out[(0,) * out.dim()]


def main():
    ws = "/data/maca-agent/agent_workdir"
    sys.path.insert(0, ws)
    ld = "/opt/conda/lib/python3.10/site-packages/torch/lib"
    os.environ["LD_LIBRARY_PATH"] = (f"{ld}:{os.environ['LD_LIBRARY_PATH']}"
                                     if os.environ.get("LD_LIBRARY_PATH") else ld)

    import torch

    dev = torch.device("cuda")
    n = 1000 * 512 * 512
    x = torch.randn(n, device=dev)
    sink = torch.zeros(1, device=dev)

    cases = {
        # the case row311 produces -- guard is genuinely free here
        "contiguous 1-D": x.clone(),
        # transpose swaps strides; flatten must copy
        "transposed (1000,512,512)": x.view(1000, 512, 512).transpose(1, 2),
        # a sliced view: cheap in memory, still non-contiguous
        "sliced view": x.view(1000, 512, 512)[:, ::2, :],
    }

    print(f"{'case':<26} {'shape':<20} {'contig':>7} "
          f"{'flatten':>9} {'index0':>8} {'sum':>8}")
    res = {}
    for name, t in cases.items():
        sink.zero_()
        flat = bench(lambda: sink.add_(t.flatten()[0]))
        idx = bench(lambda: sink.add_(one_element(t)))
        s = bench(lambda: sink.add_(t.sum()))
        res[name] = {"shape": list(t.shape), "contiguous": t.is_contiguous(),
                     "flatten_us": round(flat, 1), "index0_us": round(idx, 1),
                     "sum_us": round(s, 1)}
        print(f"{name:<26} {str(list(t.shape)):<20} {str(t.is_contiguous()):>7} "
              f"{flat:>9.1f} {idx:>8.1f} {s:>8.1f}")

    print("\n--- degenerate shapes must not crash the replacement ---")
    # The live harness already gates on numel > 0, so empty is the caller's
    # responsibility -- but a replacement must still be safe when that gate
    # passes, which it is not for a bare (0,) index on an empty tensor.
    for name, t in (("scalar (0-dim)", torch.tensor(3.14, device=dev)),
                    ("1-element 1-D", torch.ones(1, device=dev)),
                    ("empty 1-D (guarded)", torch.zeros(0, device=dev))):
        try:
            if t.numel() == 0:
                print(f"  {name:<20} skipped by the caller's numel > 0 guard")
                continue
            v = one_element(t)
            sink.add_(v)
            print(f"  {name:<20} OK, read {float(v):.4g}, "
                  f"numel={t.numel()}, dim={t.dim()}")
        except Exception as e:
            print(f"  {name:<20} FAIL: {type(e).__name__}: {e}")
            res[f"edge/{name}"] = str(e)

    print()
    measured = {k: v for k, v in res.items()
                if isinstance(v, dict) and "flatten_us" in v}
    worst = max(measured.items(), key=lambda kv: kv[1]["flatten_us"])
    print(f"worst case for the CURRENT guard: {worst[0]} at "
          f"{worst[1]['flatten_us']} us vs index0 {worst[1]['index0_us']} us")
    with open("guard_strided.json", "w") as f:
        json.dump(res, f, indent=2)
    print("-> guard_strided.json")
    return 0


sys.exit(main())
