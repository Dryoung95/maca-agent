#!/usr/bin/env python3
"""The DCE-guard sink costs a full memory pass. Find a cheap dependency.

Out so far, on row 311 (n = 262M, 1.05 GB/tensor):

  discard  (the gate)   extension 1469.8 us @ 1427 GB/s -- the copy ceiling is
                        1410 GB/s, so the kernel is NOT being eliminated
  consume  (profiling)  extension 2154.6 us, because out.sum() adds 693.6 us
                        (a full 1.05 GB pass at 1512 GB/s), fully serial

Adding a constant penalizes the FASTER kernel proportionally harder:
1.917 degrades to 1.623. profiling.py's DCE guard, added to stop the backend
eliminating work, is itself inflating every fast kernel it measures.

But the guard exists for a reason: LOOP_PLAN documents the discard form
understating eager by 2.3x and the extension by 3.7x. Dropping it is not safe.

The expensive part is `out.sum()` -- a reduction over every element. A true
data dependency only needs the consumer to READ the output; it does not need
to read all of it. This measures three cheap alternatives against both
criteria: (a) near-zero cost, and (b) still defeats DCE. DCE is detected by
whether runtime tracks problem size -- a kernel whose work is eliminated
reports a constant, so each variant is also run at n/16; if the smaller size
comes out ~constant the guard has failed.
"""
import json
import os
import sys


def main():
    ws = "/data/maca-agent/tasks/row311"
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

    import torch.nn.functional as F
    compiled = torch.compile(model)

    print(f"{'guard':<22} {'ext us':>8} {'ext GB/s':>9} "
          f"{'eager us':>9} {'speedup':>8} {'vs cmp':>7}  n/16 ok?")
    results = {}

    for label, scale in (("full", 1000 * 512 * 512),
                         ("n/16", 1000 * 512 * 512 // 16)):
        torch.manual_seed(42)
        x = torch.randn(scale, device=dev)
        outsink = torch.zeros(1, device=dev)

        def call(mdl):
            with torch.no_grad():
                return mdl(x)

        # Each guard: same kernel, same inputs, only the dependency differs.
        def guard_none():
            _ = call(new)

        def guard_full():
            outsink.add_(call(new).sum())

        def guard_stride():
            outsink.add_(call(new).flatten()[::4096].sum())

        def guard_scalar():
            outsink.add_(call(new).flatten()[0])

        def guard_block():
            outsink.add_(call(new).flatten()[:4096].sum())

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
            return sum(ts) / len(ts) * 1e3

        eager = bench(lambda: call(model))
        cmp = bench(lambda: call(compiled))
        for gname, gfn in (("discard", guard_none), ("sum() all", guard_full),
                           ("[::4096].sum()", guard_stride),
                           ("[0] scalar", guard_scalar),
                           ("[:4096].sum()", guard_block)):
            us = bench(gfn)
            gbs = 2 * scale * 4 / 1e9 / (us / 1e6)
            results.setdefault(gname, {})[label] = {
                "ext_us": round(us, 1), "gbs": round(gbs, 0),
                "speedup": round(eager / us, 3),
                "vs_compile": round(cmp / us, 3)}
        results["discard"][label]["eager_us"] = round(eager, 1)
        results["discard"][label]["compile_us"] = round(cmp, 1)

    for gname, d in results.items():
        f, s = d["full"], d["n/16"]
        # A guard that failed to defeat DCE shrinks super-linearly with n
        # (runtime stops tracking work). full/n16 of ~16 = scales with n.
        ratio = f["ext_us"] / s["ext_us"]
        ok = "YES" if 10 < ratio < 24 else "**NO**"
        print(f"{gname:<22} {f['ext_us']:>8.1f} {f['gbs']:>9.0f} "
              f"{d['full'].get('eager_us', 0):>9.1f} {f['speedup']:>8.3f} "
              f"{f['vs_compile']:>7.3f}  {ok} ({ratio:.1f}x)")

    with open("guard_cost.json", "w") as f:
        json.dump(results, f, indent=2)
    print("-> guard_cost.json")
    return 0


sys.exit(main())
