#!/usr/bin/env python3
"""Regression test for the cheap DCE guard on the axpby example.

probe_guard_cost.py found, on row 311, that every candidate guard defeats DCE
(each scales 12-13x from n/16 to full n) but their cost spans two orders of
magnitude: out.sum() adds a full memory pass and drags the speedup from 1.91
to 1.31, while a single scalar read is invisible.

That was ONE kernel. Before touching utils/profiling.py, the cheap guard has to
hold on the kernel the docs' reference numbers come from -- axpby -- because
those numbers are what SKILL.md's "Measured reference points" cite. If the
scalar guard fails to defeat DCE here, or changes axpby's numbers, the fix is
not safe.

axpby is also the case LOOP_PLAN documents as the DCE victim (eager
understated 2.3x, extension 3.7x), so this is the stress test that matters.
"""
import json
import os
import sys


def main():
    ws = "/data/maca-agent/agent_workdir"
    sys.path.insert(0, ws)
    ld = "/opt/conda/lib/python3.10/site-packages/torch/lib"
    os.environ["LD_LIBRARY_PATH"] = (f"{ld}:{os.environ['LD_LIBRARY_PATH']}"
                                     if os.environ.get("LD_LIBRARY_PATH") else ld)

    import torch
    import cuda_extension

    dev = torch.device("cuda")

    def bench(fn, trials=20):
        for _ in range(10):
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

    rows = []
    # n spans the docs' range: 128 is where LOOP_PLAN saw the DCE artifact,
    # 1M+ is where the reference points live.
    for n in (128, 1 << 14, 1 << 20, 1 << 22):
        torch.manual_seed(0)
        a = torch.randn(n, device=dev)
        b = torch.randn(n, device=dev)
        sink = torch.zeros(1, device=dev)

        eager = bench(lambda: torch.add(torch.mul(a, 2.0), b))
        cm = torch.compile(lambda x, y: torch.add(torch.mul(x, 2.0), y))
        _ = cm(a, b)
        cmp = bench(lambda: cm(a, b))

        r = {"n": n, "eager_us": round(eager, 3), "compile_us": round(cmp, 3)}

        # DCE check: an eliminated kernel's runtime stops tracking n. Compare
        # against n/64; a live kernel scales ~64x, a dead one ~1x.
        a2 = torch.randn(n // 64, device=dev)
        b2 = torch.randn(n // 64, device=dev)
        small = bench(lambda: cuda_extension.axpby_v2_forward(a2, b2, 2.0, 0))

        for gname, gfn in (
                ("discard", lambda: cuda_extension.axpby_v2_forward(a, b, 2.0, 0)),
                ("sum_all", lambda: sink.add_(
                    cuda_extension.axpby_v2_forward(a, b, 2.0, 0).sum())),
                ("scalar", lambda: sink.add_(
                    cuda_extension.axpby_v2_forward(a, b, 2.0, 0).flatten()[0]))):
            us = bench(gfn)
            r[gname] = {"us": round(us, 3),
                        "vs_compile": round(cmp / us, 3),
                        "vs_eager": round(eager / us, 3),
                        # live => ratio near 64; DCE'd => ratio near 1
                        "scales": round(us / small, 1)}
        rows.append(r)
        print(f"n={n:>9d} eager={eager:8.3f} compile={cmp:8.3f}")
        for gname in ("discard", "sum_all", "scalar"):
            d = r[gname]
            dce = "DCE?" if d["scales"] < 40 else ""
            print(f"   {gname:9s} {d['us']:8.3f}us  vs_cmp={d['vs_compile']:6.2f}"
                  f"  scales={d['scales']:5.1f} {dce}")
        print()

    with open("guard_axpby.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("-> guard_axpby.json")
    return 0


sys.exit(main())
