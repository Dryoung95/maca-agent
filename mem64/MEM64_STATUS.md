# 64 GB C500: memory re-derivation

**Date**: 2026-10-13
**Scope**: the task pool, memory guardrails, and documented hardware constants
were all built for a 16.3 GB C500. The card in this host is the 64 GB variant
(`torch.cuda.get_device_properties(0).total_memory` = 68.28 GB = 63.59 GiB;
`mx-smi` reports 65536 MiB with 416 MiB reserved for the driver), so every
16.3 GB decision was re-derived from measurement rather than from the static
estimator.

The environment, toolchain, compile/verify/profile chain, and the KernelBench
gate were confirmed working first — clean rebuild 23.76 s, verification 5/5,
gate `compiled=True correctness=True` on MetaX C500. Nothing below changes any
of that; this only redoes the memory sizing.

## Headline

The 16.3 GB line was a capacity artifact, not a device limit. **Every task
that the old pool excluded for being too big actually fits on this card.**

| | 16.3 GB pool (committed) | 64 GB pool (`mem64/actionable_tasks_64gb.csv`) |
|---|---|---|
| rows | 3396 | **3448** |
| `direct` | 3156 | 3448 |
| `downsize` (batch shrunk to fit) | 240 | **0** |
| excluded as over-capacity | 36 | 35 restored |
| excluded by the 10 GB admission bar | 22 | 22 restored |
| OOM observed at original batch | — | **0** across 292 measured tasks |

Reconciliation: 3396 − 5 dataset defects + 35 restored + 22 admission-gap = 3448.

## Method

`dataset/mem_budget.py` gives a static estimate per task (input + param bytes,
×3 for input + ref output + new output). LOOP_PLAN E3 already showed it runs
2–15× conservative on C500 because it cannot see forward-pass intermediates.
Admission to a memory-limited pool therefore cannot be decided by the
estimate; it needs the real peak.

Each task was executed at its **original** (un-shrunk) batch constant and the
peak read with `torch.cuda.max_memory_allocated()`. Guardrails preserved for
the shared card: `mem_get_info` before every task, 8 GB free minimum, one task
at a time, `empty_cache()` between tasks.

| script | what it measures | result |
|---|---|---|
| `probe_downsize.py` | sample of the downsize set, ascending est | 24/24 fit, 0 OOM |
| `probe_downsize.py --order desc` | the largest estimates (stress the ceiling) | the ~100 GB static estimates peak at 34.36 / 16.40 GB |
| `probe_excluded.py` | the 36 over-16.3 GB tasks | 35 fit (1 defect) |
| `sweep_downsize.py` | **all 240** downsize tasks at original batch | 233 fit, 5 defects, **0 OOM**, 26 min |
| `gap probe` | the 22 tasks in the 10–16.3 GB admission gap | 22/22 fit |
| `build_pool64.py` | assembles the pool from the measurements | `actionable_tasks_64gb.csv` |

Full-sweep numbers (240 tasks, original batch): 233 measured with a peak, 5
defects, 2 skipped for exceeding the sweep's own 95 GB ceiling. Those two
(rows 645, 1074) were then run by the ceiling probe at 34.36 / 16.40 GB and
both land in the pool — which is why 233 fit in the sweep but the pool holds
235 reverted tasks.

- measured peak: min 2.05, median 6.51, **max 25.84 GB**
- static estimate / actual: min 1.0×, median 3.0×, max 6.0×
- 0 OOM

The 48 GB admission bar leaves ≥15 GB for the other agents on this shared
card; the largest measured task used 34.36 GB, so the bar is not what excludes
anything.

## What this changes downstream

1. **No task needs its batch shrunk.** The 240 `downsize` rows revert to their
   original batch (`note=reverted_to_original_batch`). This is the biggest
   win: those tasks were designed for the paper's 1–100 ms runtime window, and
   shrinking them moved many out of it.
2. **57 previously-excluded tasks become usable** — 35 over-capacity + 22 from
   the admission gap.
3. **`MIN_FREE_GB = 9.0` in `dataset/collect_rollouts.py` is no longer the
   binding constraint.** It was sized so a 16.3 GB card could hold one episode
   (~8.66 GB measured peak) plus overhead. With 68.28 GB the achievable
   concurrency is several episodes; the guard should be raised rather than
   removed, since the card is still shared. That file is owned by the
   path-plumbing work and is left untouched here.
4. **ROADMAP's compute conclusions built on 16.3 GB no longer hold**: the
   "1.5B / 7B-QLoRA fits, full 7B needs 4–5 cards" sizing, and the
   recommendation to collect rollouts on 8–16 cards, were both derived from
   the 16.3 GB budget.

## Dataset defects (not capacity)

Six tasks fail identically at the shrunk and the original batch — an index
tensor constructed on CPU inside `get_inputs`/`forward`. This is the same
class the old pool already dropped (LOOP_PLAN E2b, row 38), not a C500 limit:
`rows 38, 199, 2278, 3075, 3096, 3160`. Verified by running each at both batch
values and matching the error. They are reported by the build script and
excluded from the pool.

Of the six, five were re-measured in this pass (rows 38, 2278, 3075, 3096,
3160) and all five are in `mem64_downsize_sweep.json`. Row 199 was **not**
re-measured here — it is carried over from the probe of the over-16.3 GB
excluded set (`mem64_excluded_probe.json`), where it failed with the same
device-mismatch error and was therefore never in the old pool. The two sources
agree on the failure mode, so the count stands at six; it is just not six
measurements from one sweep.

## Two profiling artifacts (and the row 311 discrepancy, resolved)

`utils/profiling.py` and the KernelBench gate disagree on row 311 —
1.539 vs 1.905 for the same episode, same kernel, same inputs. That gap is now
reproduced and explained, and the explanation is **not** the one recorded here
earlier. There are two artifacts, they are orthogonal, and they bite at
opposite ends of the size range.

### Artifact 1 — the timer (small n only)

`utils/profiling.py` sums `torch.profiler` **device time** over 10 iterations
in one context; the gate times each trial with start/end **cuda events**. At
small n the event number includes host-side work the device time excludes, so
they disagree by up to 8× for `torch.compile` (10.9 µs device vs 89.8 µs
event, same call):

| n | device-time ratio | cuda-event ratio | agree? |
|---|---|---|---|
| 128 | 0.55× | 2.98× | NO |
| 1,048,576 | 1.23× | 2.02× | NO |
| 8,388,608 | 1.86× | 1.89× | yes |
| 33,554,432 | 2.13× | 2.15× | yes |
| 134,217,728 | 2.22× | 2.23× | yes |
| 1,073,741,824 | 2.27× | 2.28× | yes |

("agree" = within 15%.) This part of the earlier write-up is correct, and it
is a small-n property — the two timers converge at n ≳ 8M.

### Artifact 2 — the DCE-guard sink (large n) — the real cause of 1.539 vs 1.905

Row 311 is n = 262M, far past the 8M line above, so the timer cannot explain
its 24% gap. A 2×2 was run on the actual compiled kernel
(`tasks/row311`, `fused_leaky_relu`) varying **timer** × **cache policy** —
the gate clears L2 before every trial, `profiling.py` runs warm:

| cell | ext/eager |
|---|---|
| profiler + warm | 1.628 |
| profiler + cold | 1.579 |
| cuda_event + warm | 1.624 |
| **cuda_event + cold (the gate)** | **1.619** |

All four agree within 1.6% — with the copy baseline at 1410 GB/s, i.e. the
card idle. **Neither the timer nor the cache is the cause.** Running the
official gate itself reproduced the teammate's figure exactly (ref 2.82 ms /
ext 1.48 ms = 1.905), so both numbers are real.

The remaining difference is what each path does with the output. The gate does
`_ = kernel_fn(*args)` — **discards**. `profiling.py` does
`sink.add_(out.sum())` — a full global reduction, added to defeat DCE.
Measured at n = 262M, same kernel, same inputs, cuda events:

| guard | ext µs | ext GB/s | ext/eager |
|---|---|---|---|
| discard (the gate) | 1479.0 | 1419 | **1.917** |
| `flatten()[0]` scalar | 1480.6 | — | **1.924** |
| `.sum()` (the form at the time) | 2164.1 | 969 | **1.630** |

The reduction costs 685 µs in isolation — a full extra 1.05 GB pass at
1512 GB/s. It is strictly additive and serial: discard 1469.8 + 693.6 = 2163.4
predicted vs 2154.6 measured (eager: 2817.7 + 693.6 = 3511.3 vs 3498.4). So
`profiling.py`'s own DCE guard is what produces 1.539, and a **constant**
penalizes the faster kernel proportionally harder — 24% of eager, 46% of the
extension.

### Is the sink protecting anything?

**At row 311 scale, no.** Discard and the scalar guard agree to within 0.5%
across all three targets at n = 262M, and the discard kernel still runs at
1419 GB/s against a 1410 GB/s copy ceiling — so there is no elimination to
guard against on the model-forward path. Probed across n = 128 / 1024 / 65 536
on the axpby example, no target ever shows the DCE signature
(discard < 0.70× scalar) at any size.

**At small n the guard is not free either — a caveat this section originally
missed.** Measured across the range (`probe_guard_scale.py`), the one-element
read costs 6–15 µs at n ≤ 4M and converges to the discard timing only near
n = 8M (table below). So the fix does not remove the artifact class, it moves
it: `sum()` cost a constant 685 µs at large n, `flatten()[0]` costs a constant
~8 µs at small n. Both shrink the measured ratio toward 1, in the false-fail
direction.

The sink was added for a real reason — LOOP_PLAN documents the discard form
understating eager by 2.3× and the extension by 3.7× — but that does not
reproduce on the current model-return paths, which were all that was tested.
The G6 case it actually fixed (an inline-lambda construction) was **not**
re-tested here, so this does not show the guard was unnecessary in general —
only that it is not load-bearing on the paths stage-1 collection will use.

**Consequence for the loop, stated plainly:** `profiling.py` and the reward
gate disagree by construction, not by noise — the gate discards the result,
`profiling.py` consumes it. The internal performance signal a rollout sees in
its feedback can therefore read `correct_slow` while the gate that sets the
reward reads `pass`, on the same kernel.

### The direction of the error (corrected: it is not one-sided)

An earlier version of this section said the profiler's speedup is a *strict
lower bound* on the gate's. That is true only for ratios above 1. The real
mechanism is a **shrink toward 1**, which cuts both ways. A guard costing C
adds C to both timings, so a true ratio r against a baseline of b comes out as
`1 + (r−1)·b/(b+C)`; the compression factor `b/(b+C) < 1` pulls *every* ratio
toward 1, whatever its sign:

- r > 1 → under-reported (false fail)
- r < 1 → over-reported

The harmful direction is exclusively the false fail, because a false pass is
provably impossible: passing needs `measured ≥ 1.05` while `r ≤ 1.05`, i.e.
`(r−1.05)·b ≥ 0.05·C`, whose left side is ≤ 0 while the right is positive. So
the sink never admits a losing kernel — it rejects winners. With C = 683 µs and
b = 1463 µs, a true 1.05× kernel reads 1.034× and fails, and a true 1.07× reads
1.048× and fails; only at true 1.08× does it clear the bar.

### Fix applied, and its own caveat

`utils/profiling.py` now reads `out.flatten()[0]` (with a `numel > 0` guard)
instead of `out.sum()` — the dependency is preserved at a fraction of the
cost, and the harness was verified after the change (verification 5/5,
`run_smoke` passes, gate `compiled=True correctness=True`).

But the one-element read is free only where work dominates launch. Measured
across n on axpby (`probe_guard_scale.py`), it costs 6–15 µs at n ≤ 4M and only
converges to the discard timing near n = 8M:

| n | ext discard | ext 1-elem | delta | compile/ext measured vs true |
|---|---|---|---|---|
| 128 | 22.1 | 30.1 | +8.0 | 2.75 vs 3.19 |
| 1,024 | 22.1 | 30.4 | +8.2 | 3.10 vs 3.58 |
| 16,384 | 22.3 | 30.0 | +7.8 | 3.17 vs 3.58 |
| 262,144 | 23.6 | 30.3 | +6.6 | 3.10 vs 3.38 |
| 4,194,304 | 59.8 | 62.8 | +3.0 | 1.83 vs 1.81 |
| 8,388,608 | 99.6 | 103.1 | +3.5 | 1.63 vs 1.61 |

(17 of 18 target/n pairs show a cost above the run-to-run spread.) On these
tasks it never produced a false fail because axpby wins by 1.3–3.6×, far above
the bar — but a marginal 1.05–1.07 kernel at n < 8M would fail.

**The one-element form is also not safe on non-contiguous outputs** — a second
defect found after the change landed. `out.flatten()[0]` materializes a
**contiguous copy** when the output is not already row-major, so it costs a full
read+write pass, the very defect the change was made to remove. Measured on a
1 GB tensor (`probe_guard_strided.py`):

| case | contiguous | `flatten()[0]` | `out[(0,)·dim]` | `sum()` |
|---|---|---|---|---|
| 1-D, 262M | yes | 24.7 µs | 23.4 µs | 697.3 µs |
| transposed (1000,512,512) | **no** | **1884.7 µs** | 26.9 µs | 696.4 µs |
| sliced view (1000,256,512) | **no** | 744.8 µs | 24.6 µs | 1041.9 µs |

On the transposed case the fix is **2.7× worse than the `sum()` it replaced**.
Row 311's kernel writes contiguous output, which is why the change looked free
there. It does fire on real tasks: 227 dataset tasks return a
transposed/permuted output from `forward()` (479 contain a layout-changing op),
reproduced on a real model output (`probe_guard_real.py`):
`flatten()[0]` 1882.8 µs vs a per-dim index 25.6 µs, both reading the identical
element. A per-dim zero index walks the tensor's own strides and never asks for
a contiguous layout:

```python
def _one_element(out):
    if out.dim() == 0:      # scalar output: out[()], not out[(0,)]
        return out[()]
    return out[(0,) * out.dim()]
```

Reported to the file owner; not applied here.

Neither form showed the classic constant-runtime DCE signature here (discard
scales 11.9 → 91.3 µs across n = 128 → 8M, i.e. 7.7×), and the small-n axpby
numbers are at the edge of interpretability — identical runs at 1×128 gave ext
14.9 / 15.1 / 22.9 µs. This says nothing about the G6 case the guard was added
for (an inline-lambda construction), which was not re-tested. What it does say:
on the paths as they stand the sink is not load-bearing for DCE prevention.

### Practical rule

The two artifacts bite at opposite ends of the size range, and the *fix* adds a
third at small n. **Stage-1 collection should stay on the sweet-spot tasks**
(the G1 candidates are row311-scale, n ≥ 262M, where the one-element guard is
effectively free) and away from the 1×128 example size, where launch overhead
dominates and neither the timer nor the guard can be read above the noise.

Under the discard (gate-like) form, the axpby extension beats `torch.compile`
comfortably across the range — 3.12× at n = 128 down to 1.87× at n = 4M — but
the margin *shrinks* with n here, because `torch.compile` itself gets faster
while the extension saturates the copy ceiling. The kernel is fine; the margin
is what a reward should not be calibrated against at one size.

## Files

```
mem64/
  probe_downsize.py        sample probe of the downsize set (--order asc/desc/spread)
  probe_excluded.py        the 36 over-16.3 GB tasks
  sweep_downsize.py        full sweep, resumable, appends after each task
  build_pool64.py          builds the pool from the probe JSONs
  actionable_tasks_64gb.csv  the result: 3448 rows, all direct
  mem64_downsize_sweep.json  all 240 downsize measurements
  mem64_excluded_probe.json  35 of the 36 over-16.3 GB tasks
  mem64_gap_probe.json       the 22 admission-gap tasks
  probe_timer.py           the timer × cache 2×2 that ruled out artifact 1
  probe_consume.py         consume-vs-discard A/B, found artifact 2
  probe_sum_cost.py        out.sum() in isolation; the additive decomposition
  probe_guard_cost.py      cheap DCE-guard candidates on row 311
  probe_guard_axpby.py     guards on axpby (bad `scales` diagnostic, superseded)
  probe_guard_scale.py     guard cost across n = 128..8M (the small-n caveat)
  probe_guard_strided.py   guard on non-contiguous outputs (the copy bug)
  probe_guard_real.py      same bug on a real model output, not a synthetic tensor
  probe_dce_model.py       same-n DCE test on the model forward, n = 262M
  probe_dce_small.py       same-n DCE test at n = 128 / 1K / 64K
  timer_2x2.json, consume_ab.json, sum_cost.json, guard_cost.json,
    dce_model.json, dce_small.json   the measurements behind the tables above
  MEM64_STATUS.md          this file
```

`dataset/` is deliberately untouched — that tree is owned by the parallel
path-plumbing work. The new pool ships here so the two stay independent until
they land together.
