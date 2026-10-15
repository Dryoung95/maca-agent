#!/usr/bin/env python3
"""Decisive test: does the profiling sink protect against anything real?

The `scales` check in probe_guard_axpby.py was mis-designed -- at n/64 every
variant is launch-bound (~22.8 us flat), so the ratio could never reach 64 and
the "DCE?" flags were false positives. The discard timings themselves DO scale
with n (22.8 -> 60.7 us), so the kernel is not being eliminated there.

That leaves the question that decides whether utils/profiling.py's `out.sum()`
sink earns its cost: is there ANY path in the profiling harness where
discarding the output actually eliminates work?

A cleaner test, which does not depend on a scaling heuristic: compare discard
against a guard whose cost is ~0 (a single scalar read, measured at ~4 us). If
the kernel is eliminated, discard lands far BELOW the scalar guard. If it runs,
discard and the scalar guard agree to within the scalar read. That is a
same-n comparison -- no launch-overhead confusion.

Run on the real model forward (not the raw op) for all three targets the
harness profiles, at a size where work dominates launch: n = 262M.
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
    compiled = torch.compile(model)

    torch.manual_seed(42)
    inputs = [x.to(dev) for x in get_inputs()]
    n = inputs[0].numel()
    print(f"n = {n} ({n*4/1e9:.2f} GB/tensor); work dominates launch here\n")

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

    print(f"{'target':<12} {'discard':>9} {'scalar':>9} {'sum_all':>9} "
          f"{'discard/scalar':>15}  verdict")
    res = {}
    for tname, fn in (("eager", lambda: model(*inputs)),
                      ("compile", lambda: compiled(*inputs)),
                      ("extension", lambda: new(*inputs))):
        sink = torch.zeros(1, device=dev)

        def discard():
            _ = fn()

        def scalar():
            out = fn()
            o = out[0] if isinstance(out, (list, tuple)) else out
            sink.add_(o.flatten()[0])

        def sum_all():
            out = fn()
            o = out[0] if isinstance(out, (list, tuple)) else out
            sink.add_(o.sum())

        d, sc, sm = bench(discard), bench(scalar), bench(sum_all)
        # discard below scalar by more than the scalar read's own ~4 us
        # means work disappeared. At n=262M that is a big signal or nothing.
        drop = sc - d
        verdict = ("DCE -- sink needed" if drop > 20
                   else "no DCE -- sink is pure overhead")
        res[tname] = {"discard": round(d, 1), "scalar": round(sc, 1),
                      "sum_all": round(sm, 1),
                      "scalar_minus_discard_us": round(drop, 1),
                      "verdict": verdict}
        print(f"{tname:<12} {d:>9.1f} {sc:>9.1f} {sm:>9.1f} "
              f"{d/sc:>14.3f}  {verdict}")

    print()
    for tname in ("eager", "compile", "extension"):
        r = res[tname]
        print(f"{tname}: sink costs {r['sum_all'] - r['discard']:.0f} us "
              f"({(r['sum_all']/r['discard'] - 1)*100:.0f}% overhead)")
    speed = {"discard": res["eager"]["discard"] / res["extension"]["discard"],
             "scalar": res["eager"]["scalar"] / res["extension"]["scalar"],
             "sum_all": res["eager"]["sum_all"] / res["extension"]["sum_all"]}
    print()
    print(f"speedup with discard  : {speed['discard']:.3f}  <- the gate")
    print(f"speedup with scalar   : {speed['scalar']:.3f}  <- proposed sink")
    print(f"speedup with sum_all  : {speed['sum_all']:.3f}  <- current "
          "utils/profiling.py")
    with open("dce_model.json", "w") as f:
        json.dump({"n": n, "targets": res, "speedup": speed}, f, indent=2)
    print("\n-> dce_model.json")
    return 0


sys.exit(main())
