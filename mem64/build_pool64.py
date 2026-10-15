#!/usr/bin/env python3
"""Rebuild the actionable task pool for the 64 GB C500 card.

WHY: dataset/actionable_tasks.csv (3396 rows) was sized for the 16.3 GB C500 --
3156 `direct` tasks, 240 `downsize` tasks whose batch constant was shrunk to
fit a 10 GB budget, and 36 over-16.3 GB tasks excluded outright. This host has
the 64 GB variant, so the shrink and the exclusion are artifacts of a wrong
budget, not device limits. LOOP_PLAN E3 showed the static estimator runs 2-15x
conservative, so admission is decided by the MEASURED peak
(max_memory_allocated at the original batch, in mem64/*_probe.json), not the
estimate. The bar is 48 GB, not 64: the card is shared and its free memory
drifts, so the pool must leave room. Unprobed or over-budget downsize tasks
keep their shrunk batch plus a note; a task the probe killed for a
non-capacity reason (index tensor left on CPU) is a dataset defect, not a C500
limit, so it is reported and dropped (LOOP_PLAN E2b, row 38).

Output: mem64/actionable_tasks_64gb.csv, written here rather than into dataset/
because that tree is owned by the path-plumbing work.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(os.path.dirname(HERE), "dataset")
BUDGET_GB = 48.0  # admission bar on the MEASURED peak; downsize was shrunk to 10
PROBE_FILES = ("mem64_downsize_probe.json", "mem64_ceiling.json",
               "mem64_excluded_probe.json")


def load_probes():
    """Merge every mem64_*.json probe here into {row: record}, including the
    one-task-per-file probes (a single record, not row-keyed)."""
    merged = {}

    def put(row, rec):
        # A measured record never loses to a later skip/defect record: the
        # ceiling probe measured rows 645/1074, then the downsize probe
        # overwrote them with bare over_ceiling skips.
        if row not in merged or ("peak_used_gb" not in merged[row]
                                 and "peak_used_gb" in rec):
            merged[row] = rec

    names = {n for n in os.listdir(HERE)
             if n.startswith("mem64_") and n.endswith(".json")}
    for name in sorted(set(PROBE_FILES) | names):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"  warning: probe {name} missing; its tasks are unprobed")
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  warning: unreadable {name} ({type(e).__name__}); "
                  "its tasks are unprobed")
            continue
        if isinstance(data, dict):
            entries = ([(int(data["row"]), data)] if "row" in data
                       else [(int(k), v) for k, v in data.items()
                             if isinstance(v, dict)])
            for row, rec in entries:
                put(row, rec)
    return merged


def main():
    import pandas as pd

    # Every probe field is read with .get(): a measured entry carries
    # peak_used_gb / exec_ok, an over-ceiling skip only ok/reason, a defect
    # only err -- a missing key is normal, not an error.
    probes = load_probes()
    peak = {r: v["peak_used_gb"] for r, v in probes.items()
            if v.get("peak_used_gb") is not None}
    # Non-capacity failures: a real error, not a capacity skip (those carry
    # ok/reason instead). Reported and dropped, never added to the pool.
    defects = sorted(
        (r, str(v.get("err") or v.get("error"))[:90])
        for r, v in probes.items()
        if (v.get("err") or v.get("error")) and not v.get("exec_ok", False))
    defect_rows = {r for r, _ in defects}

    def read_csv(name):
        path = os.path.join(DATASET, name)
        if not os.path.exists(path):
            sys.exit(f"missing required input: {path}")
        return pd.read_csv(path)

    at = read_csv("actionable_tasks.csv")
    ds = read_csv("downsize_analysis.csv").set_index("row")
    mb = read_csv("mem_budget.csv")

    cols = ["row", "strategy", "est_gb", "grade", "batch", "batch_to", "note"]
    NA = float("nan")   # plain NaN keeps the batch columns float, not object
    frames = []

    # ---- 1. direct rows carry over; batch/batch_to are downsize-only -------
    direct = at[at.strategy == "direct"].copy()
    direct[["batch", "batch_to"]] = NA
    frames.append(direct.assign(note="direct_unchanged"))

    # ---- 2. downsize rows: revert to the ORIGINAL batch where measured -----
    # at.est_gb is already the original-batch estimate, so only the batch
    # columns change.
    dz = at[at.strategy == "downsize"].copy()
    n_downsize = len(dz)
    dz["peak"] = dz.row.map(peak)
    # A defect fails at any batch size, and it only stayed in the old pool
    # because the failure surfaced later -- keep it out of the new one.
    dz = dz[~dz.row.isin(defect_rows)]
    rev = dz[dz.peak.notna() & (dz.peak <= BUDGET_GB)]
    frames.append(rev.assign(strategy="direct", batch=NA, batch_to=NA,
                             note="reverted_to_original_batch"))
    kept = dz[dz.peak.isna() | (dz.peak > BUDGET_GB)].copy()
    if len(kept):
        kept["note"] = ["unprobed_keep_downsize" if pd.isna(p)
                        else f"probed_{p:.1f}GB_over_budget" for p in kept.peak]
        frames.append(kept)

    print(f"downsize tasks: {n_downsize} "
          f"({n_downsize - len(dz)} dropped as defects)")
    print(f"  reverted to original batch (measured <= {BUDGET_GB:.0f} GB): "
          f"{len(rev)}, peak {rev.peak.min():.2f}-{rev.peak.max():.2f} GB")
    print(f"  kept shrunk (unprobed or over budget): {len(kept)}")

    # ---- 3. previously excluded, plus the 10-16.3 GB admission gap ----------
    # The old pool drew its direct line at ~10 GB (its own max is 9.87 GB), so
    # two groups never made it in:
    #   a) est > 16.3 GB   -- dropped as over-capacity (36 tasks)
    #   b) est 10-16.3 GB  -- simply below the admission bar (22 tasks)
    # Both are arbitrary on a 64 GB card, and both are measured by the same
    # probes, so they are handled identically; (b) was never oversized, only
    # below the bar. est_bytes comes from mem_budget, the old pool's own rows
    # are already counted above, so they are excluded here by row.
    mbz = mb[mb.est_bytes > 0].copy()
    mbz["est_gb"] = mbz.est_bytes / 1e9
    excluded = mbz[(mbz.est_gb > 16.3) & (~mbz.row.isin(set(at.row)))].copy()
    excluded["peak"] = excluded.row.map(peak)
    restored = excluded[excluded.peak.notna()
                        & (excluded.peak <= BUDGET_GB)].sort_values("row")
    frames.append(restored.assign(strategy="direct", batch=NA, batch_to=NA,
                                  note="restored_previously_excluded"))

    # The admission-gap set: below the old 10 GB direct bar but not in the pool
    # either. > 10.0 admits exactly this group given the 9.87 GB ceiling above.
    gap = mbz[(mbz.est_gb > 10.0) & (mbz.est_gb <= 16.3)
              & (~mbz.row.isin(set(at.row)))].copy()
    gap["peak"] = gap.row.map(peak)
    gap_ok = gap[gap.peak.notna() & (gap.peak <= BUDGET_GB)].sort_values("row")
    frames.append(gap_ok.assign(strategy="direct", batch=NA, batch_to=NA,
                                note="restored_admission_gap"))

    print(f"\npreviously excluded tasks: {len(excluded)}, restored: "
          f"{len(restored)}, peak {restored.peak.min():.2f}-"
          f"{restored.peak.max():.2f} GB")
    print(f"admission-gap tasks (10 < est <= 16.3 GB): {len(gap)}, restored: "
          f"{len(gap_ok)}"
          + (f", peak {gap_ok.peak.min():.2f}-{gap_ok.peak.max():.2f} GB"
             if len(gap_ok) else ""))
    if defects:
        print("\ntask-level failures (dataset defects, not capacity):")
        for r, err in defects:
            print(f"  row {r}: {err}")

    # ---- 4. assemble and write --------------------------------------------
    # Disjoint by construction; drop_duplicates guards against a probe file
    # double-covering a row (direct first, so it wins over downsize).
    pool = pd.concat([f[cols] for f in frames], ignore_index=True)
    pool = pool.drop_duplicates(subset="row", keep="first").sort_values("row")
    pool["row"] = pool.row.astype(int)
    out = os.path.join(HERE, "actionable_tasks_64gb.csv")
    pool.to_csv(out, index=False)

    measured = pool.row.map(peak).dropna()
    print(f"\n=== pool: {len(pool)} rows -> {out} ===")
    print(f"  old (16.3 GB budget): {len(at)} rows")
    print(f"  new (64 GB budget):   {len(pool)} rows "
          f"(+{len(pool) - len(at)} added)")
    print("  " + pool.strategy.value_counts().to_string().replace("\n", "\n  "))
    print(f"  measured peak over the pool: {measured.min():.2f}-"
          f"{measured.max():.2f} GB ({len(measured)} rows measured)")
    if len(defects):
        print(f"  excluded defects: {len(defects)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
