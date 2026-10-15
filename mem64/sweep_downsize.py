#!/usr/bin/env python3
"""Full-sweep probe of the downsize tasks at their ORIGINAL batch.

probe_downsize.py sampled this set (8 of 240 landed in the pool). The sample
is enough to establish that the static estimator runs 2-6x conservative, but
pool admission is decided per task, so every row needs its own measurement.

To keep this honest on a shared card it:
  - skips the torch.compile timing pass. The question here is memory, not
    speed; compile doubles wall time per task and buys nothing for a peak
    measurement. dataset/probe_tasks.py ran it for the runtime-window study;
    that data already exists.
  - writes after every task to an append-only JSON so an interrupted sweep
    resumes instead of redoing work
  - yields the card when free memory drops below the guard

Wall time is dominated by torch.compile, which we skip, so this is roughly
one task launch plus a few forward passes each -- minutes, not hours.
"""
import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(os.path.dirname(HERE), "dataset")

# Same admission bar the pool builder uses. Tasks at or under this are fit to
# run at their original batch on the 64 GB card with room left for the other
# agents sharing it.
BUDGET_GB = 48.0
MIN_FREE_GB = 8.0


def free_gb():
    import torch

    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def measure(code, collect_runtime=False):
    """One task: exec at eager, return measured peak.

    Never raises: a task-level failure returns an error record instead, so the
    sweep records dataset defects rather than dying on the first one.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    ns = {}
    exec(code, ns)
    m = ns["Model"](*ns["get_init_inputs"]()).eval().cuda()
    inputs = [t.cuda() for t in ns["get_inputs"]()]

    with torch.no_grad():
        m(*inputs)
    peak = torch.cuda.max_memory_allocated() / 1e9

    res = {"exec_ok": True, "peak_used_gb": round(peak, 3),
           "input_shape": tuple(inputs[0].shape) if inputs else None}

    if collect_runtime:
        for _ in range(5):
            with torch.no_grad():
                m(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                m(*inputs)
        torch.cuda.synchronize()
        res["eager_us"] = round((time.perf_counter() - t0) / 20 * 1e6, 1)

    del m, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return res


def load_progress(out_path):
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            print(f"  warn: unreadable {out_path}, restarting from scratch")
    return {}


def save_progress(out_path, data):
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, out_path)


def main():
    global torch
    import pandas as pd
    import torch

    sys.path.insert(0, HERE)
    from probe_downsize import rewrite_batch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default=os.path.join(HERE, "mem64_downsize_sweep.json"))
    ap.add_argument("--ceiling-gb", type=float, default=95.0,
                    help="skip a task whose static est exceeds this; the "
                         "sample showed est/actual >= 2x, but the two ~100 GB "
                         "estimates are close enough to the card to verify "
                         "individually")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--runtime", action="store_true",
                    help="also record eager runtime (for the 1ms-100ms window)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW measurements")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(DATASET, "data.parquet"))
    at = pd.read_csv(os.path.join(DATASET, "actionable_tasks.csv"))
    ds = pd.read_csv(os.path.join(DATASET, "downsize_analysis.csv"))
    mb = pd.read_csv(os.path.join(DATASET, "mem_budget.csv")).set_index("row")

    dz = at[at.strategy == "downsize"].copy()
    dsx = ds.set_index("row")
    dz["batch_name"] = dsx.loc[dz.row, "batch_name"].values
    dz["orig_batch"] = dsx.loc[dz.row, "batch_val"].values
    dz["per_batch_b"] = dsx.loc[dz.row, "per_batch_bytes"].values
    dz["param_b"] = mb.loc[dz.row, "param_bytes"].values
    dz["est_orig_gb"] = (dz.per_batch_b * dz.orig_batch + dz.param_b * 3) / 1e9
    dz = dz.sort_values("est_orig_gb")

    results = load_progress(args.out)
    n_new = n_ok = n_oom = n_defect = 0
    t_start = time.time()

    for _, r in dz.iterrows():
        rid = int(r.row)
        if str(rid) in results:
            continue
        if args.limit is not None and n_new >= args.limit:
            break
        if free_gb() < args.min_free_gb:
            print(f"\n[YIELD] only {free_gb():.2f} GB free — pausing the sweep")
            break
        if r.est_orig_gb > args.ceiling_gb:
            rec = {"row": rid, "ok": False, "reason": "over_ceiling",
                   "est_orig_gb": round(r.est_orig_gb, 2)}
            results[str(rid)] = rec
            save_progress(args.out, results)
            continue

        print(f"row {rid}: batch {r.batch_to:.0f} -> {r.orig_batch:.0f} "
              f"(est {r.est_orig_gb:.1f} GB) ... ", end="", flush=True)
        rec = {"row": rid, "batch_from": float(r.batch_to),
               "batch_to": float(r.orig_batch),
               "est_orig_gb": round(r.est_orig_gb, 2)}
        try:
            new_code, nrep = rewrite_batch(df.at[rid, "code"],
                                           r.batch_name, r.orig_batch)
            assert nrep >= 1, f"batch rewrite matched {nrep} sites"
            rec.update(measure(new_code, collect_runtime=args.runtime))
            n_ok += 1
            if rec["peak_used_gb"] <= BUDGET_GB:
                tag = "FITS"
            else:
                tag = f"OVER {BUDGET_GB:.0f}GB"
            print(f"peak {rec['peak_used_gb']:.2f} GB  {tag}"
                  + (f"  eager {rec.get('eager_us')}us" if args.runtime else ""))
        except Exception as e:
            kind = "OOM" if "OutOfMemory" in type(e).__name__ else type(e).__name__
            if kind == "OOM":
                n_oom += 1
            else:
                n_defect += 1
            rec.update({"exec_ok": False, "err": f"{kind}: {str(e)[:110]}"})
            print(f"FAIL {kind}: {str(e)[:60]}")

        results[str(rid)] = rec
        save_progress(args.out, results)
        n_new += 1

    n = len(results)
    print(f"\n=== sweep summary ===")
    print(f"records: {n} total ({n_new} new in this run, "
          f"{time.time() - t_start:.0f}s)")
    peaks = [v["peak_used_gb"] for v in results.values()
             if v.get("peak_used_gb") is not None]
    fits = sum(1 for p in peaks if p <= BUDGET_GB)
    print(f"measured: {len(peaks)}  fit <= {BUDGET_GB:.0f} GB: {fits}  "
          f"OOM: {n_oom}  defects: {n_defect}")
    if peaks:
        print(f"peak GB: min {min(peaks):.2f} median "
              f"{sorted(peaks)[len(peaks)//2]:.2f} max {max(peaks):.2f}")
    ratios = sorted(v["est_orig_gb"] / v["peak_used_gb"]
                    for v in results.values()
                    if v.get("peak_used_gb") is not None)
    if ratios:
        print(f"static est / actual: min {min(ratios):.1f}x median "
              f"{ratios[len(ratios)//2]:.1f}x max {max(ratios):.1f}x")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
