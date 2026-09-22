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
| GPU | MetaX C500 (104 SM, 16.3 GB, **warp size 64**) |
| MACA | 3.3.0 (`/opt/maca-3.3.0`) |
| CUDA-syntax front end | `cucc` (cu-bridge), at `/opt/maca-3.3.0/tools/cu-bridge/bin` |
| PyTorch | C500 build, conda at `/opt/conda` (Python 3.10) |
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

Also source the MACA flag set (provides `-DUSE_MACA` and the cu-bridge include
path) or `cucc` fails on `__macro_mxcc.h not found`:

```bash
# if you have the kda-maca environment available
source /data/kda-maca/env.sh
```

## Dataset

The task pool is [CUDA-Agent-Ops-6K](https://hf-mirror.com/datasets/ByteDance-Seed/CUDA-Agent-Ops-6K)
(6000 rows). It is **not** committed — download `data.parquet` into `dataset/`:

```bash
wget -O dataset/data.parquet \
  https://hf-mirror.com/datasets/ByteDance-Seed/CUDA-Agent-Ops-6K/resolve/main/data.parquet
```

`huggingface.co` is unreachable from this network; use the mirror. Everything
under `dataset/*.py` reads `dataset/data.parquet`.

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
  pick_elementwise_ast.py   AST-based sweet-spot task picker
  actionable_tasks.csv      3396 tasks verified executable on this device
ROADMAP.md            what remains, and the compute estimate for training
LOOP_PLAN.md          audit trail of every completed sub-task
PHASE1_STATUS.md      first-phase report (build + correctness loop)
```

## Measured results (as of 2026-09-16)

11 real dataset tasks built; 3 pass the official KernelBench gate:

| Task | Ops | Speedup vs eager |
|---|---|---|
| row 311 | leaky_relu × 2 | 1.58× |
| row 1573 | log + cos | 1.39× |
| row 2885 | selu + tanh | 1.36× |

All 5 G1 tasks compiled and passed correctness; the two that fail the gate are
the chained-SFU cases (relu+softplus 0.54×, softplus+leaky 0.57×), consistent
with the task typing above.

The LLM-driven episode (`run_g9.py`) completed end to end on the axpby
example: the model produced the three files itself, verification passed 5/5,
and the KernelBench gate reported `compiled=True, correctness=True` on
MetaX C500. At that problem size launch overhead dominates, so the kernel beat
eager (1.18–1.29×) without beating `torch.compile` — the 5% bar needs a
bandwidth-bound task to be discriminative.

## Known platform gotchas

- **DCE breaks profiling.** If the measured return value is discarded, the
  backend eliminates the work and reports a runtime constant w.r.t. problem
  size. This understated the eager baseline by 2.3× and the extension by 3.7×
  at one point, and every speedup derived from it was wrong. Always consume
  the output.
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

### 2. Collect rollouts (fits on the 16 GB card)

Episode collection is **compile-bound, not compute-bound** — each attempt is
one 23–28 s cucc compile plus verification and profiling, not a training
step. Measured per-episode wall time: ~67–71 s for the axpby example, ~515 s
for a two-attempt real task (row 311). Peak GPU memory for real dataset
tasks: 0.13–8.66 GB, i.e. the 16.3 GB card holds 3–4 concurrent episodes.

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

### 3. Fine-tune a small policy model (needs the larger card)

This is where the 16 GB card stops being enough.

- 1.5B full fine-tune or 7B **QLoRA** fits on 16.3 GB, but only for a
  smoke test — the training data does not exist yet (see step 1).
- Full 7B fine-tune needs ~94 GB (weights + AdamW fp32 states + gradients +
  activations), i.e. 4–5 of these cards, and that assumes the MACA
  distributed stack (`torch.distributed` / a collective library) actually
  works, which was never verified on this single-card host.
- The upstream paper does **not** release its RL training code — verified
  against the upstream repo, which contains only the agent workspace (14
  files, no `ppo`/`reinforce`/`policy` code). Anything beyond data collection
  is original work, not a port.

### What was deliberately left out

- No RL training pipeline. The paper's is not open-sourced; reproducing it is
  a separate research project, not a port.
- No multi-card verification. Only one C500 is visible on this host, so
  interconnect topology and the MACA collective stack are untested.
- No 128K-context training. KV cache for that alone approaches the whole
  16.3 GB budget.

## License / attribution

The upstream SKILL.md, workspace layout, and dataset are from the CUDA-Agent
release (arXiv:2602.24286). The MACA Supplement sections of SKILL.md and all
tooling under `dataset/` and `agent_workdir/agent_loop.py` are port-specific
additions.
