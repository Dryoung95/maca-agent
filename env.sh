#!/usr/bin/env bash
# CUDA-Agent on MetaX C500 (MACA) — source before any agent_workdir command.
#
# Reuses the KernelBench migration environment (MACA_PATH, cu-bridge CUDA_HOME,
# mxcc/cucc PATH, LD_LIBRARY_PATH) and additionally puts the agent workspace on
# sys.path so that model_new.py can `import cuda_extension`.

KERNELBENCH_ENV=/data/cuda-harness-migration/env.sh
if [ -f "$KERNELBENCH_ENV" ]; then
    source "$KERNELBENCH_ENV"
else
    echo "warn: $KERNELBENCH_ENV missing — MACA toolchain will not be on PATH" >&2
fi

# agent_workdir must be importable so model_new.py finds the built extension
export PYTHONPATH="/data/cuda-agent-migration/agent_workdir:${PYTHONPATH}"

# The C500 build of libtorch lives outside MACA's lib64; the extension .so needs
# libc10/libtorch_python at load time or every `import cuda_extension` fails.
export LD_LIBRARY_PATH="/opt/conda/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# Warp size on C500 is 64, not 32. SKILL.md's block sizes remain valid, but any
# warp-level reasoning (__shfl_sync lane counts, occupancy denominators) must
# divide by 64.
export METAX_WARP_SIZE=64
