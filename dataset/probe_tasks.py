#!/usr/bin/env python3
"""A3: probe sampled dataset tasks on C500 — executability, real memory, runtime window.

Conservative GPU discipline: probe ONE task at a time, free everything, then
check mem_get_info before the next. Aborts the batch if free GPU memory
drops below 4 GB (other agents may be using the card).
"""
import gc
import json
import time

import torch

ROWS = [2341, 3313, 5345, 1350, 1052, 5488, 3293, 4355, 3853, 3302]
MIN_FREE_GB = 4.0


def free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def run_one(code):
    ns = {}
    exec(code, ns)
    init = ns["get_init_inputs"]()
    m = ns["Model"](*init).eval().cuda()
    inputs = [t.cuda() for t in ns["get_inputs"]()]
    free, total = torch.cuda.mem_get_info()
    base_used = (total - free) / 1e9
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = m(*inputs)
    peak = torch.cuda.max_memory_allocated() / 1e9
    # runtime
    for _ in range(5):
        with torch.no_grad():
            m(*inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            m(*inputs)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / 20 * 1e6
    res = dict(
        exec_ok=True,
        eager_us=round(eager_us, 1),
        peak_used_gb=round(peak + base_used, 3),
        input_shapes=[tuple(t.shape) for t in inputs if isinstance(t, torch.Tensor)],
        param_count=sum(p.numel() for p in m.parameters()),
    )
    # torch.compile attempt (allowed to fail)
    try:
        cm = torch.compile(m)
        with torch.no_grad():
            cm(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                cm(*inputs)
        torch.cuda.synchronize()
        res["compile_us"] = round((time.perf_counter() - t0) / 20 * 1e6, 1)
        res["compile_ok"] = True
    except Exception as e:
        res["compile_ok"] = False
        res["compile_err"] = f"{type(e).__name__}: {str(e)[:80]}"
    del m, inputs
    if "cm" in dir():
        del cm
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    import pandas as pd
    df = pd.read_parquet("/data/cuda-agent-migration/dataset/data.parquet")
    results = {}
    for row in ROWS:
        if free_gb() < MIN_FREE_GB:
            print(f"[ABORT] free GPU {free_gb():.2f}GB < {MIN_FREE_GB}GB — "
                  "yielding to other agents")
            results[row] = dict(exec_ok=False, reason="aborted_low_gpu")
            break
        print(f"--- row {row} (free {free_gb():.2f}GB) ", end="")
        try:
            r = run_one(df.at[row, "code"])
            results[row] = r
            print(f"eager={r['eager_us']}us compile_ok={r.get('compile_ok')} "
                  f"peak={r['peak_used_gb']}GB params={r['param_count']}")
        except Exception as e:
            results[row] = dict(exec_ok=False, err=f"{type(e).__name__}: {str(e)[:120]}")
            print(f"FAIL {type(e).__name__}: {str(e)[:80]}")
        gc.collect()
        torch.cuda.empty_cache()
    with open("/data/cuda-agent-migration/a3_task_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
