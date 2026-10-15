#!/usr/bin/env python3
"""Settle the timer disagreement on row 311 (n = 262M, the real task size).

utils/profiling.py reports eager_speedup 1.539; the KernelBench gate reported
1.905 for the same episode. The two differ in TWO ways at once, and the earlier
write-up blamed only the first:

  timer   profiler sums torch.profiler CUDA device_time over 10 iters inside
          one context; the gate times each trial with start/end cuda events
  cache   profiling.py runs 10 back-to-back iters (warm); the gate calls
          clear_l2_cache (a 256 MB thrash) before EVERY trial (cold)

So this is a 2x2, not an A/B. Each cell is the same kernel on the same inputs;
only the timer and the cache policy change. If the gap lives in the timer
column it is the earlier story; if it lives in the cache row, the earlier
story is wrong and the doc needs correcting.

Usage: probe_timer.py <workspace>   (e.g. .../tasks/row311)
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", help="per-task workspace holding model.py")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "timer_2x2.json"))
    args = ap.parse_args()

    ws = os.path.abspath(args.workspace)
    # The extension is importable only from its own workspace (model_new.py
    # does `import cuda_extension`), so resolve it rather than chdir into it.
    sys.path.insert(0, ws)
    for p in ("/opt/conda/lib/python3.10/site-packages/torch/lib",):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{p}:{ld}" if ld else p

    import torch
    import torch.nn.functional as F
    from model import Model, get_inputs, get_init_inputs
    from model_new import ModelNew

    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")

    print(f"device: {torch.cuda.get_device_name(dev)} "
          f"L2 {torch.cuda.get_device_properties(dev).L2_cache_size/1e6:.1f} MB")

    # ---- copy probe: the card is shared and its baseline drifts. Any perf ----
    # number below is only meaningful if the ceiling is where the docs put it.
    n = 1000 * 512 * 512
    src = torch.randn(n, device=dev)
    dst = torch.empty(n, device=dev)
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for _ in range(20):
        dst.copy_(src)
    torch.cuda.synchronize()
    ts = []
    for i in range(10):
        torch.cuda.synchronize()
        ev[0].record()
        dst.copy_(src)
        ev[1].record()
        torch.cuda.synchronize()
        ts.append(ev[0].elapsed_time(ev[1]))
    copy_us = sum(ts) / len(ts) * 1e3
    # read 1 GB + write 1 GB
    copy_gbs = (2 * n * 4 / 1e9) / (copy_us / 1e6)
    print(f"copy baseline: {copy_us:.1f} us = {copy_gbs:.0f} GB/s "
          "(docs say ~1350 shared-idle, ~745 contended)")

    # ---- models ----
    model = Model(*get_init_inputs()).eval().to(dev)
    new = ModelNew(*get_init_inputs()).eval().to(dev)
    new.load_state_dict(model.state_dict())
    inputs = [x.to(dev) for x in get_inputs()]
    compiled = torch.compile(model)
    sink = torch.zeros(1, device=dev)
    print(f"problem: n = {n} = {n/1e6:.0f}M elements "
          f"({n*4/1e9:.2f} GB per tensor)\n")

    def consume(out):
        sink.add_(out.sum())

    def eager_fn():
        with torch.no_grad():
            consume(model(*inputs))

    def compile_fn():
        with torch.no_grad():
            consume(compiled(*inputs))

    def ext_fn():
        with torch.no_grad():
            consume(new(*inputs))

    # ---------------- the four cells ----------------
    def clear_l2():
        # KernelBench timing.clear_l2_cache, verbatim in effect: 256 MB thrash.
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=dev)
        dummy.fill_(42)
        del dummy

    def bench_profiler(fn, clear):
        # utils/profiling.py: N iters inside ONE profiler context, device_time
        # summed. With clear=True the thrash sits between iters, the way the
        # gate interleaves it -- isolates cache from the one-context choice.
        for _ in range(5):
            fn()
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as ctx:
            torch.cuda.synchronize()
            for _ in range(args.iters):
                fn()
                if clear:
                    clear_l2()
            torch.cuda.synchronize()
        return (sum(e.device_time for e in ctx.events()
                    if e.device_type.name == "CUDA") / args.iters)

    def bench_event(fn, clear):
        # KernelBench time_execution_with_cuda_event: warmup 3, empty_cache,
        # then per-trial events with discard_first=1.
        for _ in range(3):
            fn()
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ts = []
        for trial in range(args.iters + 1):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            if clear:
                clear_l2()
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            if trial >= 1:
                ts.append(s.elapsed_time(e))
        return sum(ts) / len(ts) * 1e3   # us

    targets = {"eager": eager_fn, "torch.compile": compile_fn,
               "extension": ext_fn}
    cells = [("profiler", bench_profiler), ("cuda_event", bench_event)]
    res = {"n": n, "copy_gbs": copy_gbs, "times": {}}
    for cname, bench in cells:
        for clear in (False, True):
            key = f"{cname}+{'cold' if clear else 'warm'}"
            row = {}
            for tname, fn in targets.items():
                us = bench(fn, clear)
                row[tname] = round(us, 2)
            row["ext_over_eager"] = round(row["eager"] / row["extension"], 3)
            row["ext_over_compile"] = round(row["torch.compile"]
                                            / row["extension"], 3)
            res["times"][key] = row
            print(f"{key:20s} eager={row['eager']:8.1f}us "
                  f"compile={row['torch.compile']:8.1f}us "
                  f"ext={row['extension']:8.1f}us "
                  f"| ext/eager={row['ext_over_eager']:.3f} "
                  f"ext/compile={row['ext_over_compile']:.3f}")

    eager = res["times"]["cuda_event+cold"]
    warm = res["times"]["profiler+warm"]
    print(f"\nthe disputed pair: profiler warm says ext/eager = "
          f"{warm['ext_over_eager']}, the gate (event cold) says "
          f"{eager['ext_over_eager']}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
