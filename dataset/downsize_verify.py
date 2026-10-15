#!/usr/bin/env python3
"""E2b: verify the linear downsize extrapolation with real GPU runs.

Rewrites the batch constant in the task code, runs it, measures actual peak
memory, and compares against the prediction. One task at a time, with the
low-free-memory guard.
"""
import gc
import re

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

MIN_FREE_GB = 4.0
TARGET_BATCH = {38: 256, 90: 256, 130: 64, 143: 256, 193: 96, 212: 160}


def free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def rewrite_batch(code, name, new_val):
    # replace the module-level assignment `name = <num>`
    pat = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)(\d+)", re.M)
    new, n = pat.subn(rf"\g<1>{new_val}", code)
    return new, n


def main():
    import pandas as pd
    df = paths.load_dataframe()
    ds = pd.read_csv(os.path.join(paths.DATASET_DIR, "downsize_analysis.csv")).set_index("row")
    mb = pd.read_csv(os.path.join(paths.DATASET_DIR, "mem_budget.csv")).set_index("row")

    results = {}
    for row, want in TARGET_BATCH.items():
        if free_gb() < MIN_FREE_GB:
            print(f"[ABORT] free={free_gb():.2f}GB — yielding to other agents")
            results[str(row)] = {"ok": False, "reason": "aborted_low_gpu"}
            break
        orig = df.at[row, "code"]
        bname = ds.at[row, "batch_name"]
        r = {"row": int(row), "target_batch": want, "batch_name": bname}
        try:
            new_code, n = rewrite_batch(orig, bname, want)
            assert n >= 1, "batch rewrite failed"
            ns = {}
            exec(new_code, ns)
            m = ns["Model"](*ns["get_init_inputs"]()).eval().cuda()
            inputs = [t.cuda() for t in ns["get_inputs"]()]
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                m(*inputs)
            actual = torch.cuda.max_memory_allocated() / 1e9
            r["actual_peak_gb"] = round(actual, 3)
            r["exec_ok"] = True
            # prediction from linear model
            per_batch = ds.at[row, "per_batch_bytes"]
            param_bytes = mb.at[row, "param_bytes"]
            pred = (per_batch * want + param_bytes * 3) / 1e9
            r["pred_gb"] = round(pred, 3)
            r["ratio"] = round(actual / pred, 2) if pred > 0 else None
            r["within_budget"] = actual <= 10.0
            print(f"row {row}: batch={want} actual={actual:.2f}GB "
                  f"pred={pred:.2f}GB ratio={actual/pred:.2f} ok={actual<=10.0}")
        except Exception as e:
            r["exec_ok"] = False
            r["err"] = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"row {row}: FAIL {type(e).__name__}: {str(e)[:80]}")
        results[str(row)] = r
        gc.collect()
        torch.cuda.empty_cache()

    import json
    json.dump(results, open(os.path.join(paths.REPO_ROOT, "e2b_downsize_verify.json"), "w"),
              indent=2)
    ok = sum(1 for v in results.values() if v.get("within_budget"))
    print(f"\nTOTAL within 10GB budget: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
