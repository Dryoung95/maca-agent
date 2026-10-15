#!/usr/bin/env python3
"""Probe the over-16.3 GB tasks that the 16.3 GB pool excluded ENTIRELY.

actionable_tasks.csv has 3396 rows = 3156 direct (static est <= 10 GB) +
240 downsize (shrunk to fit a 10 GB budget). A third group was dropped
outright: 36 tasks whose static estimate exceeds 16.3 GB and which never got a
downsize entry (no identifiable batch constant, or the batch rewrite did not
apply). 34 of the 36 have a static estimate that fits the 64 GB card.

On the 16.3 GB card excluding them was a capacity decision. On 64 GB it is
not, unless they are too big even now. This measures the real peak for the
ones with est <= 64 GB and reports how many become usable. Same guardrails as
probe_downsize: no batch rewrite needed here -- these were never rewritten.
"""
import argparse
import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATASET = os.path.join(REPO, "dataset")

CEILING_GB = 45.0
MIN_FREE_GB = 8.0


def main():
    global torch
    import pandas as pd
    import torch

    sys.path.insert(0, HERE)
    from probe_downsize import free_gb, run_task

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ceiling-gb", type=float, default=CEILING_GB)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--out", default=os.path.join(HERE, "mem64_excluded_probe.json"))
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(DATASET, "data.parquet"))
    at = pd.read_csv(os.path.join(DATASET, "actionable_tasks.csv"))
    mb = pd.read_csv(os.path.join(DATASET, "mem_budget.csv"))
    mbz = mb[mb.est_bytes > 0].copy()
    mbz["est_gb"] = mbz.est_bytes / 1e9

    # over 16.3 GB statically AND not already in the actionable pool
    over = mbz[mbz.est_gb > 16.3]
    excluded = over[~over.row.isin(set(at.row))].sort_values("est_gb")
    eligible = excluded[excluded.est_gb <= args.ceiling_gb].head(args.n)

    print(f"excluded-from-pool tasks over 16.3 GB: {len(excluded)}")
    print(f"  of those with static est <= {args.ceiling_gb:.0f} GB: "
          f"{int((excluded.est_gb <= args.ceiling_gb).sum())}")
    print(f"  probing {len(eligible)}\n")

    results = {}
    n_ok = n_oom = n_fail = 0
    max_seen = 0.0
    for _, r in eligible.iterrows():
        rid = int(r.row)
        free = free_gb()
        if free < args.min_free_gb:
            print(f"[YIELD] only {free:.2f} GB free — stopping")
            break
        print(f"row {rid}: est {r.est_gb:.1f} GB, free {free:.2f} GB ... ",
              end="", flush=True)
        rec = {"row": rid, "est_gb": round(r.est_gb, 2)}
        try:
            res = run_task(df.at[rid, "code"])
            rec.update(res)
            if res["exec_ok"]:
                n_ok += 1
                max_seen = max(max_seen, res["peak_used_gb"])
                rec["est_over_actual"] = round(r.est_gb / res["peak_used_gb"], 1)
                print(f"peak {res['peak_used_gb']:.2f} GB "
                      f"(est/actual {rec['est_over_actual']}x) "
                      f"eager {res['eager_us']}us "
                      f"compile={res.get('compile_ok')}")
        except Exception as e:
            kind = "OOM" if "OutOfMemory" in type(e).__name__ else type(e).__name__
            if kind == "OOM":
                n_oom += 1
            else:
                n_fail += 1
            rec.update(dict(ok=False, err=f"{kind}: {str(e)[:120]}"))
            print(f"FAIL {kind}: {str(e)[:70]}")
        results[str(rid)] = rec
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== probe summary ===")
    print(f"excluded tasks probed: {n_ok} ok, {n_oom} OOM, {n_fail} other fail")
    print(f"highest actual peak seen: {max_seen:.2f} GB")
    ratios = [v["est_over_actual"] for v in results.values()
              if v.get("est_over_actual")]
    if ratios:
        s = sorted(ratios)
        print(f"static estimate over actual: min {min(ratios)}x "
              f"median {s[len(s)//2]}x max {max(ratios)}x")
    print(f"-> {args.out}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
