#!/usr/bin/env python3
"""RFT positive-sample filter for the stage-1 G1 pool.

Judgment hierarchy (strictest first):

1. OFFICIAL — tasks re-run under the official KernelBench protocol
   (cuda_event, fp32 1e-4, reference = eager) on 2026-10-15. This is the
   arbiter; see gate_recheck_all.log. A task is a win iff gate correctness
   is True AND effective speedup >= 1.0.
2. NOT_RECHECKABLE — the correct artifact was overwritten on disk by a
   later attempt, so no official verdict exists. Excluded from positives
   (we cannot verify them) but counted separately.
3. Everything else — internal numbers only, NOT used for positives.

The internal `compile_speedup` (vs torch.compile) is deliberately NOT used
as a positive criterion: torch.compile on this device frequently regresses
below eager, so beating it does not mean winning. Row 311 is the proof —
0.95x internal (negative) vs 1.76x official (win). Conversely row 480 read
1.38x internal and 0.77x official.

The internal `eager_speedup >= 1.0` field is kept as a TRIAGE signal only
(100% recall / 62% precision against the official verdict on the 16
rechecked tasks). It is never the final word.

Usage:
    python3 dataset/rft_filter.py [--rollouts rollouts/g1_pilot.jsonl]
                                  [--out rollouts/rft_positives.jsonl]
"""
import argparse
import json
import os

# ---- Officially re-measured 2026-10-15 (gate correctness True) ----
# row -> effective speedup vs eager (runtime/ref_runtime)
OFFICIAL_WINS = {
    # 12 bar-passers rechecked
    325: 2.09, 2885: 1.72, 2919: 1.68, 2793: 1.09,
    424: 0.78, 480: 0.77, 494: 0.72, 163: 0.85, 1384: 0.66, 557: 0.42,
    # 16 non-bar correct tasks rechecked (all gate-correct)
    55: 1.78, 128: 0.55, 147: 0.89, 196: 1.23, 203: 0.69, 288: 1.11,
    311: 1.76, 320: 0.64, 604: 1.76, 735: 0.65, 792: 1.91,
    1120: 0.64, 1222: 0.59, 1833: 1.84, 2081: 1.77, 2491: 0.65,
}
# Bar-passers that FAILED gate correctness (max_diff > official 1e-4).
OFFICIAL_CORRECTNESS_FAIL = {299, 2871}
# Correct on disk was overwritten by a later, incorrect attempt; no verdict.
NOT_RECHECKABLE = {60, 162, 216, 372, 5579}

BAR = 1.05  # internal compile bar, reported for contrast only


def classify(row):
    """Return (bucket, note) for one rollout row."""
    task = row["dataset_row"]
    correct = bool(row.get("correct"))
    cs = row.get("compile_speedup") or 0.0
    es = row.get("eager_speedup") or 0.0

    if not correct:
        return "negative", "incorrect kernel"
    if task in OFFICIAL_CORRECTNESS_FAIL:
        return "negative", "gate correctness fail (fp32 1e-4)"
    if task in OFFICIAL_WINS:
        eff = OFFICIAL_WINS[task]
        if eff >= 1.0:
            return "positive", f"official {eff:.2f}x vs eager"
        return "negative", f"official {eff:.2f}x vs eager (slower)"
    if task in NOT_RECHECKABLE:
        return "unverifiable", "correct artifact overwritten on disk"
    return "unverified", "no official recheck; internal only"


def pick_positive_row(task_rows):
    """Pick the row that corresponds to the artifact on disk.

    The gate measured tasks/rowN/model_new.py, which is the LAST attempt's
    file. Among correct rows we therefore take the last one — that is the
    code the official verdict describes.
    """
    correct_rows = [r for r in task_rows if r.get("correct")]
    if not correct_rows:
        return None
    return correct_rows[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="rollouts/g1_pilot.jsonl")
    ap.add_argument("--out", default="rollouts/rft_positives.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.rollouts)
            if l.strip()]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["dataset_row"], []).append(r)

    buckets = {"positive": [], "negative": [], "unverifiable": [],
               "unverified": []}
    for task, trs in sorted(by_task.items()):
        row = pick_positive_row(trs)
        if row is None:
            row = trs[-1]
        bucket, note = classify(row)
        rec = dict(row)
        rec["filter_bucket"] = bucket
        rec["filter_note"] = note
        buckets[bucket].append(rec)

    n_pos = len(buckets["positive"])
    print(f"tasks: {len(by_task)}  rows: {len(rows)}")
    for k, v in buckets.items():
        print(f"  {k:13s} {len(v):3d} tasks")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for rec in buckets["positive"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {n_pos} positives -> {args.out}")

    print("\npositive tasks (official verdict):")
    for rec in sorted(buckets["positive"], key=lambda r: -OFFICIAL_WINS[r["dataset_row"]]):
        t = rec["dataset_row"]
        print(f"  row{t:>5}  {OFFICIAL_WINS[t]:.2f}x   "
              f"attempt={rec.get('attempt')}  turns={rec.get('turn', 0)}")

    print("\nwarnings:")
    for rec in buckets["unverifiable"]:
        print(f"  row{rec['dataset_row']}: {rec['filter_note']}")


if __name__ == "__main__":
    main()
