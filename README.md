# CUDA-Agent — MetaX C500 / MACA Port

Port of the agent environment from [CUDA-Agent](https://arxiv.org/abs/2602.24286)
(ByteDance Seed / Tsinghua AIR / SIA-Lab) to the MetaX C500 GPU with the MACA
software stack.

This repo covers the **skill-augmented agent environment**: the workspace
layout, the SKILL.md constraints, the compile/verify/profile toolchain, and a
programmatic agent-loop driver. It does not include the RL training pipeline
(see [ROADMAP.md](ROADMAP.md)).

## Hardware / software requirements

These are the versions this was built and verified against:

| Component | Version |
|---|---|
| GPU | MetaX C500 (104 SM, **warp size 64**); this host has the 64 GB variant — C500 also ships as 16.3 GB, and the memory guards read the device rather than assuming either |
| MACA | 3.3.0.15 (`/opt/maca-3.3.0`, symlinked as `/opt/maca`) |
| CUDA-syntax front end | `cucc` (cu-bridge), at `/opt/maca/tools/cu-bridge/bin` |
| PyTorch | 2.8.0+metax3.3.0.2, C500 build, conda at `/opt/conda` (Python 3.10) |
| KernelBench | source tree on `PYTHONPATH` for the official eval gate |

There is **no `nvcc`**. `torch.utils.cpp_extension` is wired to `cucc`, which
translates CUDA syntax to the C500 target (xcore1000).

## Setup

```bash
source env.sh
```

This puts the MACA toolchain on `PATH`, sets `CUDA_HOME` to the cu-bridge
directory, and adds two things the port needs:

- `PYTHONPATH` must contain the workspace, or `model_new.py` cannot
  `import cuda_extension` when KernelBench loads it in an isolated context.
- `LD_LIBRARY_PATH` must contain the C500 libtorch directory, or every
  `import cuda_extension` fails with `libc10.so: not found`.

It also exports `MACA_CUCC_FLAGS` (provides `-DUSE_MACA` and the cu-bridge
include path) or `cucc` fails on `__macro_mxcc.h not found`.

KernelBench is expected at `/data/maca-harness/KernelBench/src`; override with
`KERNELBENCH_SRC` if it lives elsewhere. `env.sh` resolves its own location, so
the repo works wherever it is checked out.

## Dataset

The task pool is [CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
(6000 rows). It is **not** committed — download `data.parquet` into `dataset/`:

```bash
curl -L -o dataset/data.parquet \
  https://hf-mirror.com/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K/resolve/main/data.parquet
```

`huggingface.co` is unreachable from this network; use the mirror. Everything
under `dataset/*.py` reads `dataset/data.parquet` via `dataset/paths.py`.

Note the org is `BytedTsinghua-SIA`, the same as the [code repo](https://github.com/BytedTsinghua-SIA/CUDA-Agent).
An earlier version of this README linked `ByteDance-Seed/...`, which 404s.

## Running the example

```bash
cd agent_workdir
bash utils/compile.sh              # ~23 s
python3 -m utils.verification      # correctness 5/5
python3 -m utils.profiling         # eager / torch.compile / extension
```

The example is `axpby` (`alpha * a + b`). Its measured reference points are in
[agent_workdir/SKILL.md](agent_workdir/SKILL.md) under "Measured reference
points".

## The agent loop

`agent_loop.py` drives the SKILL.md UNIFIED WORKFLOW programmatically:

```
Step 1  implementation   write kernels/*.cu + *_binding.cpp + model_new.py
Step 2  compile + verify utils.compile, utils.verification (5/5 required)
Step 3  profile          utils.profiling (DCE-safe)
Step 4  iterate          until correctness passes, then push performance
Step 5  cleanup          kernels/ keeps only the final version
```

It is LLM-agnostic: pass a `produce_attempt` callable and the same harness
drives a scripted baseline or a model rollout. The reward follows the paper's
milestone scheme (compile_failed 0.0 → compiled_but_incorrect 0.1 →
correct_unmeasured 0.3 → correct_slow 0.5 → pass 1.0), gated at SKILL.md's 5%
speedup over `torch.compile`.

Verify the harness itself before trusting any rollout:

```bash
python3 run_smoke.py    # feeds a known-good axpby attempt end to end
```

### Driving it with a real model (stage 0)

`llm_backend.py` implements the `produce_attempt` contract with an LLM: it
hands the model the full SKILL.md and `model.py`, and requires one
`write_files` tool call producing the three files. `run_g9.py` drives a full
episode — the model writes files, the harness compiles / verifies / profiles /
gates them, and on failure the compile stderr, verification diff, or
performance gap goes back to the model for a retry (up to 3 attempts).

```bash
python3 run_g9.py                  # one LLM episode on the axpby example
```

The rollout trace is written to `g9_report.json` for later RFT filtering.
Platform-specific failure modes (warp 64, shared 64 KB, `../binding_registry.h`,
dtype normalization, grid granularity, chained-SFU dependency) are in the
system prompt so the model is constrained the same way a rollout from the
paper's agent is.

## Two baselines, do not conflate them

SKILL.md's 5% bar is measured against **torch.compile**. The KernelBench
official gate (`eval_kernel_against_ref`) measures against **eager**. A
memory-bound fused kernel can beat eager comfortably and still lose to
torch.compile — both ratios are recorded per attempt (`eager_speedup` and the
compile ratio) so a downstream consumer never reads one as the other.

A third axis: the two paths consume the output differently, so the same kernel
gives different speedups from each even against the *same* baseline. The gate
discards the result; `profiling.py` reads one element of it. Whatever the guard
costs is added to both timings, which shrinks every measured ratio toward 1 —
under-reporting winners and over-reporting losers. That makes a false pass
impossible and a false fail the only failure mode. Details in the DCE gotcha
below.

## What is winnable on this device

Full table with measured evidence is in SKILL.md ("Task typing"). Summary:

| Task shape | Verdict |
|---|---|
| Memory-bound elementwise fusion, float output | **Sweet spot** — fuse aggressively |
| Activation chains with single-SFU or branch-free ops | **Winnable** with float2 vectorization |
| Chained SFU dependency (`expf`→`log1pf`, e.g. softplus) | **Hard** — the dependency chain starves the scheduler |
| Reductions / convolutions / attention | **Hard wall** — cuDNN owns these |
| bool / int8 output | **Deprioritize** — non-coalesced stores |

Grid granularity matters: a scalar kernel at one thread per element reached
only ~600 GB/s; the same math in `float2` with a fixed 1024-block grid-stride
loop reached ~1300 GB/s against a ~1350 GB/s copy ceiling.

## Repository layout

```
agent_workdir/        the ported environment (SKILL.md + tools + agent loop)
  SKILL.md            upstream workflow + MACA Supplement (port-specific notes)
  agent_loop.py       programmatic driver for the 5-step workflow
  run_smoke.py        harness self-test
  llm_backend.py      LLM implementation of the produce_attempt contract
  run_g9.py           one LLM-driven episode (stage 0)
  utils/              compile.sh / compile.py / verification.py / profiling.py
dataset/              task-pool tooling (does not include data.parquet —
                      download it separately, see below)
  paths.py                 shared path resolution; every other script imports
                           this instead of hardcoding absolute paths
  collect_rollouts.py      batch LLM episode collection → (prompt, response,
                           reward) JSONL; append-only so runs resume
  pick_elementwise_ast.py   AST-based sweet-spot task picker
  actionable_tasks.csv      3396 tasks verified executable on this device
ROADMAP.md            what remains, and the compute estimate for training
LOOP_PLAN.md          audit trail of every completed sub-task
PHASE1_STATUS.md      first-phase report (build + correctness loop)
mem64/                the 64 GB re-derivation: measurable pool build,
                       probes, and write-up (see below)
```

## The 64 GB card

The 16.3 GB figures throughout this repo come from the C500 variant the port
was originally built on. This host has the **64 GB** variant
(`get_device_properties(0).total_memory` = 68.28 GB = 63.59 GiB), and
re-measuring from that changes the task pool: the batch-shrinking and the
exclusion of "too big" tasks turn out to be artifacts of the smaller board,
not device limits.

Every task the old pool excluded for capacity was run at its **original**
batch size, with peak memory measured via `max_memory_allocated`:

| | 16.3 GB pool | 64 GB pool |
|---|---|---|
| rows | 3396 | 3448 |
| tasks with batch shrunk to fit | 240 | 0 |
| excluded as over-capacity | 36 | 35 restored |
| dropped by the ~10 GB admission bar | 22 | 22 restored |
| OOM at original batch | — | 0 (292 tasks measured) |

Measured peaks: min 2.05 / median 6.51 / max 25.84 GB; the static estimator
runs median 3.0× conservative, which is why admission is measured rather than
estimated. Six tasks are excluded for a non-capacity reason — an index tensor
constructed on CPU — verified by failing identically at the shrunk and the
original batch (row 199 is carried over from LOOP_PLAN E2b; only 5 were
re-measured here). All detail is in `mem64/MEM64_STATUS.md`.

The 22 admission-gap rows sit at est 10–15.1 GB, so they were excluded by the
old pool's ~10 GB *direct* bar, not the 16.3 GB capacity line — an earlier
`build_pool64.py` filtered on `est_gb > 16.3` and silently missed them. The
filter is `> 10.0` now, and the shipped CSV reconciles: 3448 unique row ids
(`3156 direct_unchanged` + `235 reverted_to_original_batch` +
`22 restored_admission_gap` + `35 restored_previously_excluded`).

The memory guards read the device rather than assuming either variant, so
existing tooling still runs; the parameter that changes is `--min-free-gb`.

## Measured results (as of 2026-09-16)

11 real dataset tasks built; 3 pass the official KernelBench gate:

| Task | Ops | Speedup vs eager |
|---|---|---|
| row 311 | leaky_relu × 2 | 1.91× |
| row 1573 | log + cos | 1.39× |
| row 2885 | selu + tanh | 1.36× |

Row 311's figure was previously listed as 1.58×; that number came through
`utils/profiling.py`, whose `out.sum()` DCE guard costs a full memory pass.
The official gate (ref 2.82 ms vs ext 1.48 ms, re-run 2026-10-13) gives
**1.905**. The two paths disagree by construction, not noise — see the DCE
gotcha above and `mem64/MEM64_STATUS.md`. Both figures beat eager, so the guard
shrinks this one toward 1; 1.58 was conservative, not wrong.

Rows 1573 and 2885 were not re-measured — their workspaces are gone (Step 5
keeps only the final kernel, and only `tasks/row311/` survived), so their
figures come through the same path and are shrunk the same way. They passed the
gate, which the no-false-pass property guarantees is unaffected; only the
margin is uncertain.

All 5 G1 tasks compiled and passed correctness; the two that fail the gate are
the chained-SFU cases (relu+softplus 0.54×, softplus+leaky 0.57×), consistent
with the task typing above.

The LLM-driven episode (`run_g9.py`) completed end to end on the axpby
example: the model produced the three files itself, verification passed 5/5,
and the KernelBench gate reported `compiled=True, correctness=True` on
MetaX C500. At that problem size launch overhead dominates, so the kernel beat
eager (1.18–1.29×) without beating `torch.compile` — the 5% bar needs a
bandwidth-bound task to be discriminative.

**Stage-1 rollout collection is unblocked and complete on the G1 pool**
(2026-10-14). 51 tasks, 136 rollout rows, all complete (prompt, response,
reward) triples, via `dataset/collect_rollouts.py`:

| | count |
|---|---|
| tasks collected | 51 / 51 |
| rollout rows | 136 |
| correct kernels | 57 |
| tasks passing SKILL.md's 1.05× bar vs torch.compile | **12** |
| KernelBench official gate passed | 12 |

Best results are 2.10× (softplus+ReLU), 2.04× (ReLU+log1p), 1.42×
(tanh+mish) — all beating `torch.compile` comfortably. Every
`nn.Softmax`/`LogSoftmax`/`Softmin` task (6) produced 0 correct kernels,
matching the task typing: reductions are not winnable by elementwise fusion,
and those rows are legitimate negative samples.

**Three endpoint bugs had to be fixed before any of this was collectable** —
see the gotchas below. Per-episode wall time is ~4–9 min; before the fixes it
was 45–104 min with zero output.

Reading these numbers, note that ~11 tasks were collected before the
cross-attempt feedback loop was fixed (the compile stderr was silently
dropped, see G10 in LOOP_PLAN.md), so their 0-correct results understate the
model. The 12 passes all came after that fix.

## Known platform gotchas

- **The LLM endpoint enables extended thinking by default, and it is
  invisible.** Behind `ANTHROPIC_BASE_URL=https://ai.gitee.com/anthropic`
  (Atria-Dawn-Preview), the model with thinking on burns the *entire* output
  budget on hidden reasoning: the stream contains no `content_block_*` events
  at all, yet `output_tokens` hits the max and `stop_reason=max_tokens`, so no
  `write_files` call is ever emitted. This presents exactly like "max_tokens
  too small" — 8192 truncates, 16384 truncates, raising the limit fixes
  nothing. The fix is `thinking={"type": "disabled"}`; with it off, row 3579
  needs 2623 output tokens, not >16k. See
  `agent_workdir/llm_backend.py::call_model`.

- **A hard ~600s cutoff on any connection to that endpoint**, streaming or
  not (measured three times: 609/622/607s, as `RemoteProtocolError` or a 502).
  It is not an idle timeout — content was flowing. Any single call that
  would take >10 min is uncompletable regardless of retries. Streaming is
  also ~5× faster independently of the cutoff (non-streaming buffers the
  whole generation at the gateway, ~7–16 tok/s; streaming runs ~22–97 tok/s),
  so `call_model` streams.

- **DCE breaks profiling, but the common fix overshoots.** If the measured
  return value is discarded, the backend can eliminate the work and report a
  runtime constant w.r.t. problem size. So `utils/profiling.py` consumes the
  output. The original form was `sink.add_(out.sum())` — correct in principle
  but expensive: `out.sum()` is a full extra memory pass (693 µs at n = 262M,
  i.e. 1.05 GB), strictly serial. It now reads `out.flatten()[0]`, one element,
  which keeps the dependency at a fraction of the cost. Measured on row 311:
  `sum()` → ext/eager 1.63×, one-element → 1.92×, discard → 1.92×.

  The mechanism is worth knowing because it is **bidirectional**, not a one-way
  conservative bias. A guard that costs C adds C to both timings, so a true
  ratio r measured against a baseline b comes out as `1 + (r−1)·b/(b+C)` —
  every ratio is shrunk toward 1 by the factor `b/(b+C) < 1`:

  - r > 1 (kernel beating its baseline) → **under-reported** → false fail
  - r < 1 (kernel losing) → **over-reported**

  The dangerous case is a false pass, and that is provably impossible: passing
  needs `measured ≥ 1.05` while `r ≤ 1.05`, and `(r−1)·b ≥ 0.05·C` cannot hold
  when the left side is ≤ 0 and the right is positive. So the sink can only
  ever reject real winners, never admit real losers — true r = 1.05–1.07 lands
  at 1.03–1.05 and fails the bar.

  Two caveats on the fix, both in the false-fail direction:
  - The one-element read is free only where work dominates. It costs 6–15 µs
    at n ≤ 4M (launch-bound) and converges to the discard timing near n = 8M,
    compressing the ratio by up to 0.48 on axpby at small n.
  - **`flatten()[0]` is not safe on non-contiguous outputs** and this is the
    more serious one. `.flatten()` materializes a contiguous copy when strides
    are not row-major, so on a 1 GB transposed tensor it costs 1884 µs against
    27 µs for a per-dim `out[(0,)·out.dim()]` index — 2.7× worse than the
    `sum()` it replaced. 227 dataset tasks return a transposed/permuted output
    from `forward()`, so it fires on real workloads; row 311's kernel writes
    contiguous output, which is why it looked free there. A per-dim index walks
    the tensor's own strides and never asks for a contiguous layout (`out[()]`
    for a 0-dim output). Stage-1 G1 tasks are row311-scale and contiguous, so
    this does not block collection, but it needs fixing before any task whose
    forward transposes.

  Details, the full 2×2 (timer × cache policy), and both size/layout tables are
  in `mem64/MEM64_STATUS.md`.
- **`ops` metadata is unreliable.** The dataset's `ops` field records only
  `torch.*` calls and misses `nn` modules constructed in `forward` — a task
  listed as `["torch.log","torch.sigmoid"]` can contain an `nn.Linear`. Use
  AST parsing of the source instead.
- **GPU is shared.** The copy baseline drifts between ~1350 and ~745 GB/s
  when other workloads run. Any performance judgment must first confirm the
  copy baseline is stable, or the measurement is noise.
- **`TORCH_CUDA_ARCH_LIST=9.0` from the upstream SKILL.md must not be used**
  — it is Hopper-specific. Leave unset; PyTorch auto-selects xcore1000.

## Next steps

The environment itself is complete and verified. What is not done is **using**
it at scale. The order below is not optional — each step is the precondition
for the next, and the first one is cheap.

### 1. Persist complete rollout traces (small, do this first)

`llm_backend.py` currently records only metadata per model exchange
(`turn`, `stop_reason`, `produced`). It does **not** store the prompt text, the
model's full output, or the tool-call arguments — so every episode is
consumed and discarded. RFT / SFT needs (prompt, response, reward) triples;
right now zero such records exist. This is the tail end of the environment
port, not new work.

**What to do**: extend `make_attempt_from_llm` to append the full user prompt,
the raw assistant content blocks, and the extracted `write_files` input to
`trace`, alongside the reward the harness computes for that attempt.

### 2. Collect rollouts (this card is roomy — 64 GB)

Episode collection is **compile-bound, not compute-bound** — each attempt is
one 23–28 s cucc compile plus verification and profiling, not a training
step. Measured per-episode wall time: ~67–71 s for the axpby example, ~515 s
for a two-attempt real task (row 311). Peak GPU memory for real dataset
tasks: 0.13–8.66 GB, so this card holds several concurrent episodes; the
previously quoted "3–4" assumed the 16.3 GB variant. The memory guard in
`collect_rollouts.py` reads the device, so raising `--min-free-gb` is the way
to exploit the extra headroom, not a change to the tasks.

`dataset/collect_rollouts.py` drives this. It does not require the caller to
have sourced `env.sh` — it puts the workspace and kernelbench on `sys.path`
itself. Output is append-only JSONL, so an interrupted run resumes:

```bash
# the G1 sweet-spot pool, verified by AST to be pure elementwise
python3 dataset/collect_rollouts.py --from-csv dataset/g1_candidates.csv \
    --n 10 --out rollouts.jsonl

# or specific rows
python3 dataset/collect_rollouts.py --rows 311,1573,2885 --out rollouts.jsonl
```

Each row is one (prompt, response, reward) triple for a model turn that
produced files. Verified on row 311: 2 rows, both positive, KernelBench gate
`compiled=True correctness=True`.

- Task pool: 3396 executable tasks (`dataset/actionable_tasks.csv`).
- Use the G1 sweet-spot filter (`dataset/pick_elementwise_ast.py`) to select
  first; G1 measured 3/5 tasks passing the official gate, so expect roughly
  60% yield.
- **Check the copy probe before any performance judgment** — this GPU is
  shared and its baseline drifts between ~1350 and ~745 GB/s. A measurement
  taken while another workload holds the card is noise.
- Known non-winners to deprioritize: chained-SFU dependencies (`expf`→
  `log1pf`), bool/int8 outputs (non-coalesced stores), conv / attention
  (cuDNN owns these).

### 3. Fine-tune a small policy model (still needs the larger card)

This is where the single card stops being enough, and 64 GB changes the
numbers but not the conclusion.

- A 1.5B full fine-tune fits comfortably on the 64 GB board; 7B **QLoRA**
  does too, and full 7B is now roughly one card short. The real blocker is
  unchanged: the training data does not exist yet (see step 1).
- Full 7B fine-tune needs ~94 GB (weights + AdamW fp32 states + gradients +
  activations), i.e. 2–3 of these cards, and that assumes the MACA
  distributed stack (`torch.distributed` / a collective library) actually
  works, which was never verified on this single-card host.
- The upstream paper does **not** release its RL training code — verified
  against the upstream repo, which contains only the agent workspace (14
  files, no `ppo`/`reinforce`/`policy` code). Anything beyond data collection
  is original work, not a port.

The 4–5 cards this section previously quoted were sized for the 16.3 GB
variant; on the 64 GB board the same workload needs 2–3. Both numbers assume
`torch.distributed` works on MACA, which is still unverified.

### What was deliberately left out

- No RL training pipeline. The paper's is not open-sourced; reproducing it is
  a separate research project, not a port.
- No multi-card verification. Only one C500 is visible on this host, so
  interconnect topology and the MACA collective stack are untested.
- No 128K-context training. KV cache for that alone approaches the whole
  16.3 GB budget (and would now leave room for little else even at 64 GB).

## License / attribution

The upstream SKILL.md, workspace layout, and dataset are from the CUDA-Agent
release (arXiv:2602.24286). The MACA Supplement sections of SKILL.md and all
tooling under `dataset/` and `agent_workdir/agent_loop.py` are port-specific
additions.
