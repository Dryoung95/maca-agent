#!/usr/bin/env python3
"""Does the sink's own reduction explain the timer disagreement?

probe_consume.py showed the mechanism: `discard` gives the gate's 1.9, `consume`
gives profiling.py's 1.6. Both forms DO run the kernel at full size -- the
extension discards at 1427 GB/s, essentially the copy ceiling, so this is not
the DCE the repo documents ("a discarded return lets the backend eliminate
work and report a runtime constant w.r.t. problem size").

The consume form adds work: `sink.add_(out.sum())` is a global reduction over
262M elements. If that reduction is cheap relative to the kernel, the consume
timings should equal the discard timings plus the reduction. If it is NOT
serial -- if it makes the stream longer for eager and the extension
differently -- the ratio moves some other way.

This measures the reduction on its own and checks the sum.
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
            ts.append(s.elapsed_time(e))
    return sum(ts) / len(ts) * 1e3    # us


def main():
    dev = torch.device("cuda")
    n = 1000 * 512 * 512
    x = torch.randn(n, device=dev)
    torch.manual_seed(42)

    sink = torch.zeros(1, device=dev)
    out = torch.empty(n, device=dev)

    sum_only = bench(lambda: x.sum())
    add_only = bench(lambda: sink.add_(x[0]))
    print(f"out.sum() alone      : {sum_only:7.1f} us "
          f"(reads {n*4/1e9:.2f} GB at {n*4/1e9/(sum_only/1e6):.0f} GB/s)")
    print(f"sink.add_ scalar     : {add_only:7.1f} us")

    ext_discard, eag_discard = 1469.8, 2817.7
    ext_consume, eag_consume = 2154.6, 3498.4
    print()
    print(f"extension  discard  : {ext_discard:7.1f} us")
    print(f"  + out.sum()       : {ext_discard + sum_only:7.1f} us  "
          f"(consume measured {ext_consume})")
    print(f"eager      discard  : {eag_discard:7.1f} us")
    print(f"  + out.sum()       : {eag_discard + sum_only:7.1f} us  "
          f"(consume measured {eag_consume})")
    print()
    pred = (eag_discard + sum_only) / (ext_discard + sum_only)
    print(f"speedup discard+sum : {pred:.3f}   (consume measured 1.624)")
    # A non-serial story: the reduction only costs its own time when the
    # kernel output it reduces is already resident in L2 -- which it is not
    # for 1 GB.
    print()
    print(f"out.sum() / ext_discard = {sum_only/ext_discard:.3f}  "
          f"--> the reduction is {sum_only/ext_discard*100:.0f}% of the "
          "fused kernel's runtime")
    print(f"out.sum() / eag_discard = {sum_only/eag_discard:.3f}  "
          f"--> and {sum_only/eag_discard*100:.0f}% of eager's, so it drags "
          "the FASTER kernel proportionally harder")
    with open("sum_cost.json", "w") as f:
        json.dump({"sum_us": sum_only, "add_us": add_only,
                   "predicted_consume_speedup": pred,
                   "measured_consume_speedup": 1.624}, f, indent=2)
    return 0


sys.exit(main())
