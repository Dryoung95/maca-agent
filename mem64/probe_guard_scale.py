#!/usr/bin/env python3
"""Is the 1-element guard actually free? (data-a1's small-n caveat)

utils/profiling.py now reads out.flatten()[0] instead of out.sum(), on the
strength of probe_dce_model.py: at n=262M the 1-elem read matched discard to
0.5% while sum() cost a full pass. That was ONE size.

data-a1 reports the guard is NOT free at small n: on axpby at n=128-262K the
read costs +5 to +11 us (read/discard 1.43-1.66x), converging only at n=8M.
If so, the change just traded a large-n artifact for a small-n one, and the
SKILL.md bar is calibrated against torch.compile -- where ext_us is the
denominator, so an added constant C compresses r>1 downward (false fail) and
r<1 upward (false pass, provably impossible).

This measures the same three targets across the full n range, using a same-n
A/B (no scaling assumptions, which is what made probe_guard_axpby.py's `scales`
diagnostic wrong before). Also records the noise floor at each size: data-a1
notes identical axpby runs at 1x128 gave 14.9/15.1/22.9 us, so the small end
may be uninterpretable.
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
    import torch.nn.functional as F
    import cuda_extension

    dev = torch.device("cuda")

    def bench(fn, trials=20, reps=5):
        """Us, median over reps (median is what a noisy launch-bound timer
        needs; mean drags on the 22.9 us outlier in a 14.9 us run)."""
        out = []
        for _ in range(reps):
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
                    ts.append(s.elapsed_time(e) * 1e3)
            out.append(sum(ts) / len(ts))
        return sum(out) / len(out), (max(out) - min(out))

    print(f"{'n':>10} {'target':<10} {'discard':>9} {'1-elem':>9} "
          f"{'delta':>7} {'d/e':>5} {'spread':>7}  verdict")
    res = []
    for n in (128, 1024, 1 << 14, 1 << 18, 1 << 22, 1 << 23):
        torch.manual_seed(0)
        a = torch.randn(n, device=dev)
        b = torch.randn(n, device=dev)
        sink = torch.zeros(1, device=dev)

        cm = torch.compile(lambda x, y: torch.add(torch.mul(x, 2.0), y))
        _ = cm(a, b)

        # custom op in-place so the SAME tensor is written either way and the
        # only difference is the read
        tmp = torch.empty_like(a)

        targets = {
            "eager": lambda: torch.add(torch.mul(a, 2.0), b, out=tmp),
            "compile": lambda: cm(a, b),
            "extension": lambda: cuda_extension.axpby_v2_forward(a, b, 2.0, 0),
        }
        for tname, fn in targets.items():
            disc, sp1 = bench(fn)
            if tname == "compile":
                # torch.compile can return a fresh tensor; match the custom
                # path by making the consumer read it
                def one_elem():
                    sink.add_(fn().flatten()[0])
            else:
                def one_elem():
                    sink.add_(fn().flatten()[0])

            one, sp2 = bench(one_elem)
            delta = one - disc
            ratio = one / disc
            # meaningful only if the added cost exceeds the run-to-run spread
            noisy = abs(delta) < sp1 + sp2
            verdict = ("noise" if noisy else
                       "guard costs real time" if delta > 0 else "guard free")
            res.append({"n": n, "target": tname, "discard": round(disc, 2),
                        "one_elem": round(one, 2),
                        "delta_us": round(delta, 2),
                        "ratio": round(ratio, 3),
                        "spread_us": round(max(sp1, sp2), 2), "noisy": noisy})
            print(f"{n:>10} {tname:<10} {disc:>9.2f} {one:>9.2f} "
                  f"{delta:>+7.2f} {ratio:>5.2f} {max(sp1,sp2):>7.2f}  {verdict}")
        print()

    with open("guard_scale.json", "w") as f:
        json.dump(res, f, indent=2)
    clean = [r for r in res if not r["noisy"] and r["delta_us"] > 0]
    print(f"rows where the 1-elem read costs more than noise: {len(clean)}/{len(res)}")
    if clean:
        small = [r for r in clean if r["n"] <= 1 << 18]
        large = [r for r in clean if r["n"] > 1 << 18]
        print(f"  at n <= 256K: {len(small)} rows, "
              f"delta {min(r['delta_us'] for r in small):.1f}-"
              f"{max(r['delta_us'] for r in small):.1f} us")
        print(f"  at n  > 256K: {len(large)} rows, "
              f"delta {min(r['delta_us'] for r in large):.1f}-"
              f"{max(r['delta_us'] for r in large):.1f} us")
    print("-> guard_scale.json")
    return 0


sys.exit(main())
