#!/usr/bin/env python3
"""Isolate the last standing difference between the two row-311 numbers.

The official gate now reproduces the teammate's figure exactly:
ref_runtime 2.82 ms / runtime 1.48 ms = 1.905. probe_timer.py reproduced the
profiling.py side (1.62-1.63). So both numbers are real and the disagreement
is not a measurement flake -- something about HOW each path runs the model.

The 2x2 ruled out the timer and the L2 policy. The remaining difference is
what the two do with the OUTPUT:

  gate (timing.time_execution_with_cuda_event):  _ = kernel_fn(*args)
      the result is DISCARDED
  profiling.py:                                  sink.add_(out.sum())
      the result is CONSUMED

This is the repo's own documented gotcha ("DCE breaks profiling": a discarded
return lets the backend eliminate work). But 1.48 ms is not a no-op, so the
elimination is partial. This script runs the SAME prebuilt extension, the SAME
inputs, under the SAME cuda-event timing, varying only consume-vs-discard, to
see whether that alone moves eager and the extension by different amounts --
which is what would make one path read 1.9 and the other 1.6.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "consume_ab.json"))
    args = ap.parse_args()

    ws = os.path.abspath(args.workspace)
    sys.path.insert(0, ws)
    ld = "/opt/conda/lib/python3.10/site-packages/torch/lib"
    os.environ["LD_LIBRARY_PATH"] = (f"{ld}:{os.environ['LD_LIBRARY_PATH']}"
                                     if os.environ.get("LD_LIBRARY_PATH") else ld)

    import torch
    from model import Model, get_inputs, get_init_inputs
    from model_new import ModelNew

    dev = torch.device("cuda")
    model = Model(*get_init_inputs()).eval().to(dev)
    new = ModelNew(*get_init_inputs()).eval().to(dev)
    new.load_state_dict(model.state_dict())

    torch.manual_seed(42)          # the gate's default seed_num
    inputs = [x.to(dev) for x in get_inputs()]
    n = inputs[0].numel()
    print(f"n = {n} ({n*4/1e9:.2f} GB/tensor)")

    def clear_l2():
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=dev)
        dummy.fill_(42)
        del dummy

    # gate timing, verbatim in policy: warmup 3, empty_cache, per-trial events,
    # clear_l2 before every trial, discard_first = 1. `apply` is the only knob.
    def bench(fn, apply):
        sink = torch.zeros(1, device=dev)

        def call():
            out = fn()
            if apply == "discard":
                return
            if isinstance(out, torch.Tensor):
                sink.add_(out.sum())
            else:
                for o in out:
                    sink.add_(o.sum())

        for _ in range(3):
            call()
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ts = []
        for trial in range(args.trials + 1):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            clear_l2()
            s.record()
            call()
            e.record()
            torch.cuda.synchronize()
            if trial >= 1:
                ts.append(s.elapsed_time(e))
        return sum(ts) / len(ts)       # ms

    # traffic: eager does two passes (r 1GB + w 1GB) x2 = 4 GB; the fused
    # extension does one pass = 2 GB. Bandwidth shows whether each is near
    # the copy ceiling or being throttled by something else.
    TRAFFIC = {"eager": 4 * n * 4, "extension": 2 * n * 4}
    fns = {"eager": lambda: model(*inputs), "extension": lambda: new(*inputs)}
    res = {"n": n, "cells": {}}
    for name, fn in fns.items():
        for mode in ("consume", "discard"):
            ms = bench(fn, mode)
            gbs = (TRAFFIC[name] / 1e9) / (ms / 1e3)
            res["cells"][f"{name}/{mode}"] = round(ms, 4)
            print(f"{name:10s} {mode:8s} {ms*1e3:8.1f} us  {gbs:6.0f} GB/s")
        res["cells"][f"{name}_ratio"] = round(
            res["cells"][f"{name}/discard"] / res["cells"][f"{name}/consume"], 3)

    e = res["cells"]
    print()
    print(f"eager   discard/consume: {e['eager_ratio']}")
    print(f"ext     discard/consume: {e['extension_ratio']}")
    print(f"speedup consume  = {e['eager/consume']/e['extension/consume']:.3f}")
    print(f"speedup discard  = {e['eager/discard']/e['extension/discard']:.3f} "
          "<-- the gate's form")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
