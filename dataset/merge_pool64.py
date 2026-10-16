#!/usr/bin/env python3
"""Merge the 64 GB pool into the actionable task list — NON-DESTRUCTIVELY.

The board in this machine is 64 GB (68.3 GB total). mem64/ re-derived the
pool for that capacity: 3448 rows vs the 3396-row list that was sized for
the 16.3 GB variant. Accounting (mem64/MEM64_STATUS.md):

    3396 − 5 dataset defects + 35 restored over-capacity
                               + 22 restored admission-gap = 3448
    240 downsize rows all reverted to their original batch (0 kept)

The 5 rows present only in the 16 GB list (38, 2278, 3075, 3096, 3160) are
CPU-index-tensor defects and are EXCLUDED, not carried over.

Ownership note: dataset/actionable_tasks.csv is consumed by
dataset/collect_rollouts.py (data-a1's file), and MEM64_STATUS.md records
that the merge was deliberately deferred rather than done unilaterally. So
this script writes NEW files only:

    dataset/actionable_tasks_merged.csv   (the merged 3448-row pool)
    dataset/g1_candidates_v2.csv          (sweet-spot candidates from it)

Nothing existing is overwritten. Swapping the canonical pool over to the
merged one is a deliberate one-line change for the file owner.

Note on est_gb: the mem64 CSV still carries the OLD static estimates
(conservative by 2–15×; the corrected measurements live only in the
mem64 measurement JSONs), so est_gb here is used only for ordering, never
as a memory budget.
"""
import argparse

import pandas as pd

OLD = "dataset/actionable_tasks.csv"
NEW = "mem64/actionable_tasks_64gb.csv"
DEFECT_ROWS = {38, 2278, 3075, 3096, 3160}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--out", default="dataset/actionable_tasks_merged.csv")
    args = ap.parse_args()

    old = pd.read_csv(args.old)
    new = pd.read_csv(args.new)
    print(f"old pool: {len(old)} rows ({len(set(old['row']) - set(new['row']))} "
          f"rows unique to it)")
    print(f"64gb pool: {len(new)} rows ({len(set(new['row']) - set(old['row']))} "
          f"rows unique to it)")

    only_old = sorted(set(old["row"]) - set(new["row"]))
    only_new = sorted(set(new["row"]) - set(old["row"]))
    print(f"  only in old (16gb): {only_old}")
    print(f"  only in new (64gb): {len(only_new)} rows, e.g. {only_new[:12]}...")

    # The 5 old-only rows must be exactly the known defects.
    assert set(only_old) == DEFECT_ROWS, (
        f"expected old-only rows {sorted(DEFECT_ROWS)}, got {only_old}")

    # 64gb pool is authoritative on the 64 GB board: reverted batches,
    # restored tasks. Keep its rows as-is (including its `note` column).
    merged = new.copy()

    # Sanity: every shared row should be direct in the merged pool, since
    # downsizing is unnecessary at 64 GB. Differing strategy/batch pairs
    # are exactly the reverted ones.
    shared = old.merge(new, on="row", suffixes=("_old", "_new"))
    diff = shared[shared["strategy_old"] != shared["strategy_new"]]
    print(f"shared rows: {len(shared)}, reverted strategy in merged: "
          f"{len(diff)} (expected 235, ±0)")

    print(f"\nmerged pool: {len(merged)} rows "
          f"({sum(merged['strategy'] == 'direct')} direct)")
    merged.to_csv(args.out, index=False)
    print(f"wrote -> {args.out}")

    # Sweet-spot count from the merged pool for a quick comparison.
    direct = merged[merged["strategy"] == "direct"]
    print(f"direct rows available to the AST picker: {len(direct)}")


if __name__ == "__main__":
    main()
