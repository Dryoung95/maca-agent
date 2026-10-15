#!/usr/bin/env python3
"""Batch rollout collection on the 16.3 GB C500 card.

Episode collection is compile-bound (23-28 s of cucc per attempt), not
compute-bound, so a single card is the right machine for this job. Peak
memory measured on real dataset tasks is 0.13-8.66 GB; this guards with a
free-memory check before starting each episode instead of trusting the
static estimates, which were shown to be off by 4-15x (see LOOP_PLAN G4).

Each episode writes one JSONL row per model turn that produced files, in the
(prompt, response, reward) shape run_g9.to_training_samples produces. Output
is appended so a run can be interrupted and resumed.

Usage:
  python3 collect_rollouts.py --rows 311,1573,2885 --out rollouts.jsonl
  python3 collect_rollouts.py --from-csv g1_candidates.csv --rows 5 --out rollouts.jsonl
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import paths

MIGRATION = paths.REPO_ROOT
AGENT_WORKDIR = paths.AGENT_WORKDIR

# This script must not depend on the caller having sourced env.sh. The
# kernelbench import in agent_loop needs that path on sys.path, and without
# the workspace on the path model_new.py cannot import cuda_extension.
_KERNELBENCH_SRC = paths.require_kernelbench_src()
for _p in (AGENT_WORKDIR, _KERNELBENCH_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Episode peak memory measured on real tasks: 0.13-8.66 GB. Require this much
# free before starting another one; the card is shared and other agents load
# it unpredictably (LOOP_PLAN G1: copy baseline drifted 1350 -> 745 GB/s).
#
# This was 9.0 flat, tuned for the 16.3 GB card where it held the pool to one
# episode at a time. C500 also ships as 64 GB (63.59 GB visible here), where
# 9.0 stops binding entirely. Keep 9.0 on the small card (no regression) but
# reserve a fifth of the card on larger ones, so concurrent collectors cannot
# starve other tenants. Overridable per run with --min-free-gb.
_TOTAL_GB = paths.total_mem_gb()
MIN_FREE_GB = 9.0 if _TOTAL_GB is None or _TOTAL_GB < 32 \
    else round(_TOTAL_GB * 0.20, 1)

# SKILL.md Step 4: at least 5% faster than torch.compile. compute_reward keys
# the milestone reward off this ratio, but reports 0.5 for everything below it
# that is still correct, so a filter needs the ratio itself.
_PASS_BAR = 1.05


# torch.compile warm-up and the extension build both allocate; this is the
# guard for a *new* episode, not the peak during one.
import torch


def free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def wait_for_gpu(min_gb, poll_s=20, max_wait_s=600):
    """Block until the card has room. The GPU is shared; a run started while
    another agent holds it produces noise, and memory-wise may OOM."""
    t0 = time.time()
    while True:
        avail = free_gb()
        if avail >= min_gb:
            return True, avail
        if time.time() - t0 > max_wait_s:
            return False, avail
        print(f"  [gpu] only {avail:.2f} GB free, waiting "
              f"(limit {min_gb} GB)...", flush=True)
        time.sleep(poll_s)


def build_task_workspace(row_id, code, out_dir):
    """Copy the pristine agent_workdir, then drop in this task's model.py.

    kernels/ and model_new.py from a previous episode must not leak into the
    next one -- a stale cuda_extension.so made eval pick up the wrong kernel
    (LOOP_PLAN G8, bug 2). copytree's ignore callback sees the *source* path,
    so pruning the previous task's kernels happens after the copy.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    def ignore_artifacts(dirpath, names):
        ign = set()
        for n in names:
            if n in ("build", "__pycache__", "cuda_extension.so"):
                ign.add(n)
        return ign

    shutil.copytree(AGENT_WORKDIR, out_dir, ignore=ignore_artifacts)

    # Per-task model.py is the *only* thing that differs between tasks.
    with open(os.path.join(out_dir, "model.py"), "w") as f:
        f.write(code)

    # Fresh kernels/ so the model starts from the task's model.py alone.
    kd = os.path.join(out_dir, "kernels")
    shutil.rmtree(kd, ignore_errors=True)
    os.makedirs(kd)

    # Leftover model_new.py / reports from a prior run on this row.
    for stale in ("model_new.py",):
        p = os.path.join(out_dir, stale)
        if os.path.exists(p):
            os.remove(p)
    for f in os.listdir(out_dir):
        if f.endswith(".json"):
            os.remove(os.path.join(out_dir, f))
    return out_dir


def run_one(row_id, code, task_dir, out_path, max_attempts, model):
    """One LLM episode. Returns (summary, run_g9_module).

    Import the harness from agent_workdir only. Reloading run_g9 after
    appending task_dir re-executes its imports from *task_dir*, which is a
    copy of agent_workdir whose own sys.path does not include kernelbench;
    agent_loop then fails to import it. The workspace copy is per-task data,
    not per-task code.
    """
    if AGENT_WORKDIR not in sys.path:
        sys.path.insert(0, AGENT_WORKDIR)
    if task_dir not in sys.path:
        sys.path.append(task_dir)
    # cuda_extension must not leak between tasks (KernelBench module cache).
    sys.modules.pop("cuda_extension", None)

    import run_g9
    try:
        summary = run_g9.run_llm_episode(task_dir, task_name=f"row{row_id}",
                                         max_attempts=max_attempts,
                                         model=model)
    finally:
        # Keep the workspace for inspection but drop the build tree; it is
        # ~100 MB of object files regenerable in one compile.
        b = os.path.join(task_dir, "build")
        if os.path.isdir(b):
            shutil.rmtree(b, ignore_errors=True)
    return summary, run_g9


def already_done(out_path, row_id):
    """Append-only output: skip rows already collected so runs resume."""
    if not os.path.exists(out_path):
        return False
    tag = f'"row{row_id}"'
    with open(out_path) as f:
        for line in f:
            if tag in line:
                return True
    return False


def parse_rows(arg):
    if arg is None:
        return None
    out = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print(f"ignoring non-numeric row: {part!r}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=None,
                    help="comma-separated dataset row ids, e.g. 311,1573")
    ap.add_argument("--from-csv", default=None,
                    help="CSV with a 'row' column; --rows picks the first N")
    ap.add_argument("--n", type=int, default=5,
                    help="with --from-csv: how many rows to take")
    ap.add_argument("--out", default="rollouts.jsonl")
    ap.add_argument("--tasks-root", default=os.path.join(MIGRATION, "tasks"))
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--dry-run", action="store_true",
                    help="build workspaces and check GPU, but call no model")
    args = ap.parse_args()

    rows = parse_rows(args.rows)
    if args.from_csv:
        with open(args.from_csv) as f:
            csv_rows = [r["row"].strip() for r in csv.DictReader(f)
                        if r.get("row", "").strip()]
        picked = [r for r in csv_rows if rows is None or r in map(str, rows)]
        rows = [int(r) for r in picked[:args.n]]
    if not rows:
        print("no rows selected (use --rows or --from-csv)")
        return 2

    import pandas as pd
    df = paths.load_dataframe()

    os.makedirs(args.tasks_root, exist_ok=True)
    ok = fail = skipped = 0
    for rid in rows:
        if rid >= len(df):
            print(f"row {rid}: out of range (dataset has {len(df)} rows)")
            fail += 1
            continue
        if already_done(args.out, rid):
            print(f"row {rid}: already in {args.out}, skipping", flush=True)
            skipped += 1
            continue

        got, avail = wait_for_gpu(args.min_free_gb)
        if not got:
            print(f"row {rid}: GPU never freed up ({avail:.2f} GB), skipping",
                  flush=True)
            fail += 1
            continue

        task_dir = os.path.join(args.tasks_root, f"row{rid}")
        code = df.iloc[rid]["code"]
        build_task_workspace(rid, code, task_dir)

        if args.dry_run:
            print(f"row {rid}: workspace ready, {free_gb():.2f} GB free "
                  f"(dry run)")
            continue

        print(f"\n=== row {rid} ({free_gb():.2f} GB free) ===", flush=True)
        t0 = time.time()
        try:
            summary, run_g9_mod = run_one(rid, code, task_dir, args.out,
                                          args.max_attempts, args.model)
        except Exception as e:
            print(f"row {rid}: episode failed: {type(e).__name__}: {e}",
                  flush=True)
            fail += 1
            continue

        samples = []
        try:
            samples = run_g9_mod.to_training_samples(summary)
        except Exception as e:
            print(f"row {rid}: sample conversion failed: {e}", flush=True)

        with open(args.out, "a") as f:
            for s in samples:
                s["dataset_row"] = rid
                f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")

        # The console line distinguishes both baselines. reward>=0.5 is
        # correct-under-the-bar, which is fine for SFT but NOT a SKILL.md pass:
        # compute_reward gives 0.5 ("correct_slow") to a kernel that lost to
        # torch.compile while beating eager by 1.9x (row 311), so reading
        # `reward` alone would over-count positives. `passed` is the real bar.
        good = [s for s in samples if (s.get("reward") or 0) >= 0.5]
        passed = [s for s in samples
                  if (s.get("compile_speedup") or 0) >= _PASS_BAR
                  and s.get("correct")]
        gate = bool(summary.get("kernelbench", {}).get("correctness"))
        print(f"row {rid}: {len(samples)} row(s), {len(good)} correct "
              f"(reward>=0.5), {len(passed)} passed bar (compile>="
              f"{_PASS_BAR:.2f}x), kernelbench={gate} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        ok += 1 if gate else 0
        if not gate:
            fail += 1

    print(f"\n=== done: {ok} passed, {fail} failed, {skipped} skipped ===")
    return 0 if ok or skipped else 1


if __name__ == "__main__":
    sys.exit(main())
