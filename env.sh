#!/usr/bin/env bash
# CUDA-Agent on MetaX C500 (MACA) — source before any agent_workdir command.
#
# Resolves its own location so the repo works wherever it is checked out.
# The historical version hardcoded /data/cuda-agent-migration and
# /data/cuda-harness-migration; both moved.

MACA_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- MACA toolchain -------------------------------------------------------
# /opt/maca is the conventional symlink to the installed SDK version.
MACA_PATH="${MACA_PATH:-/opt/maca-3.3.0}"
if [ -d /opt/maca ]; then export MACA_PATH=/opt/maca; fi
CUBRIDGE_PATH="${CUBRIDGE_PATH:-$MACA_PATH/tools/cu-bridge}"

export CUDA_HOME="$CUBRIDGE_PATH"
export PATH="$CUBRIDGE_PATH/bin:$MACA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$CUBRIDGE_PATH/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/lib64:$CUBRIDGE_PATH/lib:${LIBRARY_PATH:-}"
# cucc fails on '__macro_mxcc.h not found' without the cu-bridge include path.
export CPATH="$CUBRIDGE_PATH/include:$MACA_PATH/include:${CPATH:-}"
export MACA_CUCC_FLAGS="-DUSE_MACA -I$CUBRIDGE_PATH/include"

# ---- KernelBench ----------------------------------------------------------
# The official eval gate. Override with KERNELBENCH_SRC if it lives elsewhere.
KERNELBENCH_SRC="${KERNELBENCH_SRC:-/data/maca-harness/KernelBench/src}"
if [ ! -d "$KERNELBENCH_SRC" ]; then
    echo "warn: KernelBench source not found at $KERNELBENCH_SRC" \
         "(set KERNELBENCH_SRC) — the official eval gate will not run" >&2
fi

# ---- the two ports-specific PATH additions --------------------------------
# model_new.py imports cuda_extension, and KernelBench loads model_new.py in an
# isolated context whose cwd is *not* the workspace — both must be importable.
export PYTHONPATH="$MACA_AGENT_DIR/agent_workdir:$KERNELBENCH_SRC:${PYTHONPATH:-}"

# The C500 build of libtorch lives outside MACA's lib64; the extension .so
# needs libc10/libtorch_python at load time or `import cuda_extension` fails
# with "libc10.so: not found".
TORCH_LIB="${TORCH_LIB:-/opt/conda/lib/python3.10/site-packages/torch/lib}"
if [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

# Warp size on C500 is 64, not 32. SKILL.md's block sizes remain valid, but any
# warp-level reasoning (__shfl_sync lane counts, occupancy denominators) must
# divide by 64.
export METAX_WARP_SIZE=64

# Total board memory differs between C500 variants (16.3 GB vs 64 GB). The
# memory guardrails in the tooling read this instead of assuming a fixed card.
export METAX_TOTAL_MEM_GB="${METAX_TOTAL_MEM_GB:-}"
