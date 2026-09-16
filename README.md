# CUDA-Agent on MetaX C500 (MACA)

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
  utils/              compile.sh / compile.py / verification.py / profiling.py
dataset/              task-pool tooling (does not include data.parquet)
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

## License / attribution

The upstream SKILL.md, workspace layout, and dataset are from the CUDA-Agent
release (arXiv:2602.24286). The MACA Supplement sections of SKILL.md and all
tooling under `dataset/` and `agent_workdir/agent_loop.py` are port-specific
additions.
