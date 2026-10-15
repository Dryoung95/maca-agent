#!/usr/bin/env python3
"""Re-derive the task-pool memory budget for the 64 GB C500 card.

The task pool in dataset/actionable_tasks.csv was built for a 16.3 GB card:
3454 tasks had a static memory estimate, 161 of them exceeded 16.3 GB, and
240 "downsize" tasks had their batch constant shrunk to fit a 10 GB budget.
On the 64 GB board (torch reports 63.59 GB) that budget is arbitrary, and the
shrinking is probably unnecessary. But the static estimate was already shown
to run 4-15x conservative on this hardware (LOOP_PLAN E3), so this script
measures the real peak instead of trusting the estimate.

Approach: take the downsize tasks in ASCENDING static estimate order, restore
each to its original batch constant, run it, and record the actual peak via
max_memory_allocated(). Ascending order means the scale ramps up gradually and
the empirical ratio is learned from the cheap cases first. A ceiling on actual
usage keeps the card usable for concurrent agents -- this GPU is shared.

This script writes nothing under dataset/ (that tree is owned by the
path-plumbing work); outputs land in this directory.
"""
import argparse
import gc
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATASET = os.path.join(REPO, "dataset")

# Leave room for the other agents sharing this card. A task whose real peak
# would blow through this is skipped, not attempted.
CEILING_GB = 45.0
# Refuse to start a task when the card is this loaded by someone else.
MIN_FREE_GB = 8.0


def free_gb():
    import torch

    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def rewrite_batch(code, name, new_val):
    """Replace the module-level assignment `name = <num>` with new_val."""
    pat = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)(\d+)", re.M)
    pat = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)(\d+)", re.M)
    new, n = pat.subn(rf"\g<1>{int(new_val)}", code)
    return new, n


def run_task(code):
    """Execute one task at eager + torch.compile; report peak and runtimes.

    Mirrors dataset/probe_tasks.run_one so the numbers are comparable to the
    16.3 GB-era measurements. Returns a dict; all failures are caught by the
    caller.
    """
    import torch

    torch.cuda.reset_peak_memory_stats()
    ns = {}
    exec(code, ns)
    m = ns["Model"](*ns["get_init_inputs"]()).eval().cuda()
    inputs = [t.cuda() for t in ns["get_inputs"]()]

    with torch.no_grad():
        out = m(*inputs)
    peak = torch.cuda.max_memory_allocated() / 1e9
    shape = tuple(inputs[0].shape) if inputs else None

    # runtime window: the paper admits tasks at 1ms-100ms
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

    res = dict(exec_ok=True, peak_used_gb=round(peak, 3),
               input_shape=shape, eager_us=round(eager_us, 1),
               params=sum(p.numel() for p in m.parameters()))

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
        res["compile_us"] = round((time.perf_counter() - t0) / 20 * 1e6)
        res["compile_ok"] = True
    except Exception as e:
        res["compile_ok"] = False
        res["compile_err"] = f"{type(e).__name__}: {str(e)[:100]}"

    del m, inputs
    if "cm" in res:
        del cm
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return res


def main():
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20,
                    help="how many downsize tasks to probe (ascending static est)")
    ap.add_argument("--order", choices=["asc", "desc", "spread"], default="asc",
                    help="asc = smallest static est first (safe ramp-up); "
                         "desc = largest first (stress the capacity ceiling); "
                         "spread = evenly across the est range")
    ap.add_argument("--out", default=os.path.join(HERE, "mem64_downsize_probe.json"))
    ap.add_argument("--ceiling-gb", type=float, default=CEILING_GB)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(DATASET, "data.parquet"))
    ds = pd.read_csv(os.path.join(DATASET, "downsize_analysis.csv"))
    at = pd.read_csv(os.path.join(DATASET, "actionable_tasks.csv"))
    mb = pd.read_csv(os.path.join(DATASET, "mem_budget.csv")).set_index("row")

    # The 240 downsize tasks, with their original batch constant restored.
    dz = at[at.strategy == "downsize"].copy()
    dsx = ds.set_index("row")
    dz["batch_name"] = dsx.loc[dz.row, "batch_name"].values
    dz["orig_batch"] = dsx.loc[dz.row, "batch_val"].values
    dz["per_batch_b"] = dsx.loc[dz.row, "per_batch_bytes"].values
    dz["param_b"] = mb.loc[dz.row, "param_bytes"].values
    # static estimate at the ORIGINAL batch (what the 16.3 GB budget rejected)
    dz["est_orig_gb"] = (dz.per_batch_b * dz.orig_batch + dz.param_b * 3) / 1e9
    if args.order == "asc":
        dz = dz.sort_values("est_orig_gb")
    elif args.order == "desc":
        dz = dz.sort_values("est_orig_gb", ascending=False)
    else:
        # Evenly across the range: keeps the cheap and expensive cases
        # interleaved so a global problem shows up early rather than only at
        # the top of the distribution.
        dz = dz.sort_values("est_orig_gb")
        step = max(1, len(dz) // args.n)
        dz = dz.iloc[::step].sort_values("est_orig_gb")

    results = {}
    n_ok = n_oom = n_fail = 0
    max_seen = 0.0
    for _, r in dz.head(args.n).iterrows():
        rid = int(r.row)
        free = free_gb()
        if free < args.min_free_gb:
            print(f"[YIELD] only {free:.2f} GB free — skipping the rest")
            for _, r2 in dz[dz.est_orig_gb > r.est_orig_gb].head(0).iterrows():
                pass
            break
        if r.est_orig_gb > args.ceiling_gb:
            print(f"row {rid}: static est {r.est_orig_gb:.1f} GB exceeds "
                  f"ceiling {args.ceiling_gb:.0f} GB — skipping")
            results[str(rid)] = dict(ok=False, reason="over_ceiling",
                                     est_orig_gb=round(r.est_orig_gb, 2))
            n_fail += 1
            continue

        orig = df.at[rid, "code"]
        print(f"row {rid}: batch {r.batch_to:.0f} -> {r.orig_batch:.0f} "
              f"(est {r.est_orig_gb:.1f} GB), free {free:.2f} GB ... ",
              end="", flush=True)
        rec = {"row": rid, "batch_name": r.batch_name,
               "batch_from": float(r.batch_to),
               "batch_to": float(r.orig_batch),
               "est_orig_gb": round(r.est_orig_gb, 2)}
        try:
            new_code, nrep = rewrite_batch(orig, r.batch_name, r.orig_batch)
            assert nrep >= 1, f"batch rewrite matched {nrep} sites"
            res = run_task(new_code)
            rec.update(res)
            if res["exec_ok"]:
                n_ok += 1
                max_seen = max(max_seen, res["peak_used_gb"])
                ratio = r.est_orig_gb / res["peak_used_gb"]
                print(f"peak {res['peak_used_gb']:.2f} GB "
                      f"(est/actual {ratio:.1f}x) eager {res['eager_us']}us "
                      f"compile={res.get('compile_ok')}")
                rec["est_over_actual"] = round(ratio, 1)
                rec["in_window"] = (1000 <= res["eager_us"] <= 100_000)
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
    print(f"ran at original batch: {n_ok} ok, {n_oom} OOM, {n_fail} other fail")
    print(f"highest actual peak seen: {max_seen:.2f} GB")
    ratios = [v["est_over_actual"] for v in results.values()
              if v.get("est_over_actual")]
    if ratios:
        print(f"static estimate over actual: min {min(ratios)}x "
              f"median {sorted(ratios)[len(ratios)//2]}x max {max(ratios)}x")
    print(f"-> {args.out}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
