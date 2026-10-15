#!/usr/bin/env python3
"""C2: batch-executability probe for 20 dataset tasks.

GPU discipline: one task at a time, gc + empty_cache between, abort the batch
if free GPU memory drops below 4 GB so other agents keep the card.
"""
import gc
import json
import time

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ROWS = [2884, 3826, 4993, 1904, 5539, 1527, 2112, 2063, 3065, 4659,
        1696, 5980, 3314, 5043, 2931, 3321, 4631, 3711, 4822, 2947]
MIN_FREE_GB = 4.0


def free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def main():
    import pandas as pd
    df = paths.load_dataframe()
    results = {}
    for row in ROWS:
        if free_gb() < MIN_FREE_GB:
            print(f"[ABORT] free={free_gb():.2f}GB — yielding to other agents")
            results[str(row)] = {"exec_ok": False, "reason": "aborted_low_gpu"}
            break
        t_start = time.perf_counter()
        r = {"row": int(row)}
        try:
            ns = {}
            exec(df.at[row, "code"], ns)
            m = ns["Model"](*ns["get_init_inputs"]()).eval().cuda()
            inputs = [t.cuda() for t in ns["get_inputs"]()]
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                m(*inputs)
            r["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            # eager runtime
            for _ in range(3):
                with torch.no_grad():
                    m(*inputs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                with torch.no_grad():
                    m(*inputs)
            torch.cuda.synchronize()
            r["eager_us"] = round((time.perf_counter() - t0) / 10 * 1e6, 1)
            r["exec_ok"] = True
            try:
                cm = torch.compile(m)
                for _ in range(3):
                    with torch.no_grad():
                        cm(*inputs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    with torch.no_grad():
                        cm(*inputs)
                torch.cuda.synchronize()
                r["compile_us"] = round((time.perf_counter() - t0) / 10 * 1e6, 1)
                r["compile_ok"] = True
            except Exception as e:
                r["compile_ok"] = False
                r["compile_err"] = f"{type(e).__name__}"
            r["ops"] = df.at[row, "ops"][:60]
        except Exception as e:
            r["exec_ok"] = False
            r["err"] = f"{type(e).__name__}: {str(e)[:100]}"
        r["wall_s"] = round(time.perf_counter() - t_start, 1)
        results[str(row)] = r
        print(f"row {row:>5d}: ok={r.get('exec_ok')} eager={r.get('eager_us','-'):>8} "
              f"compile={r.get('compile_us','-'):>8} peak={r.get('peak_gb','-')}GB "
              f"({r['wall_s']}s) {r.get('ops','')}")
        gc.collect()
        torch.cuda.empty_cache()
    json.dump(results, open(os.path.join(paths.REPO_ROOT, "c2_batch_probe.json"), "w"), indent=2)
    ok = sum(1 for v in results.values() if v.get("exec_ok"))
    cok = sum(1 for v in results.values() if v.get("compile_ok"))
    print(f"\nTOTAL: eager {ok}/{len(results)}  compile {cok}/{len(results)}")


if __name__ == "__main__":
    main()
