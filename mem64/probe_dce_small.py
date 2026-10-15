#!/usr/bin/env python3
"""Does the cheap (scalar) sink defeat DCE where the docs say it bites?

probe_dce_model.py settled the large-n case (row 311, n=262M): discard and a
single scalar read agree to 0.5%, so no work is being eliminated there and
out.sum() is pure overhead -- 692 us, which penalizes the FASTER kernel
proportionally harder (24% of eager vs 46% of the extension).

But profiling.py's sink exists because LOOP_PLAN documents DCE at the axpby
example's small size (eager understated 2.3x, extension 3.7x). A fix that
silences DCE protection at the size where it is real would break the numbers
the docs are built on.

Same-n diagnostic (correct version of the earlier bad `scales` test): at a
given n, compare discard against each guard. A guard defeats DCE iff it costs
roughly the same as the full work; if the kernel is eliminated, discard lands
far below the guards. This does not depend on any scaling assumption, so it
works at n where launch overhead dominates.

Runs all three profiling targets, at both the documented-DCE size and a
mid size, on the axpby example the reference numbers come from.
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

    # torch.compile is the strongest DCE candidate: it can fuse and eliminate
    # a chain whose result is discarded, where eager and a raw extension launch
    # unconditionally.
    print(f"{'n':>10} {'target':<10} {'discard':>9} {'scalar':>9} {'sum_all':>9} "
          f"{'disc/scalar':>12} {'disc/sum':>9}")
    res = []
    for n in (128, 1024, 1 << 16):
        torch.manual_seed(0)
        a = torch.randn(n, device=dev)
        b = torch.randn(n, device=dev)
        sink = torch.zeros(1, device=dev)

        cm = torch.compile(lambda x, y: torch.add(torch.mul(x, 2.0), y))
        _ = cm(a, b)

        targets = {
            "eager": lambda: torch.add(torch.mul(a, 2.0), b),
            "compile": lambda: cm(a, b),
            "extension": lambda: cuda_extension.axpby_v2_forward(a, b, 2.0, 0),
        }
        for tname, fn in targets.items():
            def scalar():
                sink.add_(fn().flatten()[0])

            def sum_all():
                sink.add_(fn().sum())

            d, sc, sm = bench(fn), bench(scalar), bench(sum_all)
            # discard below the guards by more than noise => work eliminated,
            # and the guards are what kept it alive.
            flag = " <-- DCE here" if d < sc * 0.7 else ""
            res.append({"n": n, "target": tname, "discard": round(d, 3),
                        "scalar": round(sc, 3), "sum_all": round(sm, 3),
                        "disc_over_scalar": round(d / sc, 3),
                        "disc_over_sum": round(d / sm, 3),
                        "dce": d < sc * 0.7})
            print(f"{n:>10} {tname:<10} {d:>9.3f} {sc:>9.3f} {sm:>9.3f} "
                  f"{d/sc:>11.3f} {d/sm:>9.3f}{flag}")
        print()

    hit = [r for r in res if r["dce"]]
    print(f"rows where discard < 0.70x scalar (DCE signature): {len(hit)}")
    for r in hit:
        print(f"  n={r['n']} {r['target']}: discard {r['discard']} vs scalar "
              f"{r['scalar']} ({r['disc_over_scalar']:.2f}x)")
    if not hit:
        print("  none -- no DCE at any size on these targets")
    with open("dce_small.json", "w") as f:
        json.dump(res, f, indent=2)
    print("-> dce_small.json")
    return 0


sys.exit(main())
