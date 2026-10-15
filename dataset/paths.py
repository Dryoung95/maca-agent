#!/usr/bin/env python3
"""Shared path resolution for the maca-agent tooling.

The repo has moved between machines: it was built at
/data/cuda-agent-migration, and its dependencies (KernelBench, the MACA
toolchain) lived at /data/cuda-harness-migration. Both moved. Nothing in this
package should hold an absolute path it cannot recover from.

Every helper here resolves from the environment first (env.sh exports these)
and falls back to a candidate list, so a script works both standalone and
after `source ../env.sh`.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
AGENT_WORKDIR = os.path.join(REPO_ROOT, "agent_workdir")
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

_PARQUET = os.path.join(DATASET_DIR, "data.parquet")

# Historical locations, newest first. Kept because other machines still use them.
_KERNELBENCH_CANDIDATES = [
    os.environ.get("KERNELBENCH_SRC"),
    "/data/maca-harness/KernelBench/src",
    "/data/cuda-harness-migration/KernelBench/src",
]


def kernelbench_src():
    """KernelBench's src dir, or None. agent_loop imports kernelbench.eval from
    here; it is the official correctness/perf gate."""
    for c in _KERNELBENCH_CANDIDATES:
        if c and os.path.isdir(os.path.join(c, "kernelbench")):
            return c
    return None


def require_kernelbench_src():
    p = kernelbench_src()
    if p is None:
        raise SystemExit(
            "KernelBench source not found. Set KERNELBENCH_SRC or source "
            "../env.sh — kernelbench.eval is the official eval gate.")
    return p


def parquet_path():
    """The CUDA-Agent-Ops-6K task pool (6000 rows). Not committed; downloaded
    by the setup step."""
    return _PARQUET


def load_dataframe():
    import pandas as pd
    if not os.path.exists(_PARQUET):
        raise SystemExit(
            f"{_PARQUET} not found — download the task pool first:\n"
            f"  curl -L -o {_PARQUET} "
            "https://hf-mirror.com/datasets/BytedTsinghua-SIA/"
            "CUDA-Agent-Ops-6K/resolve/main/data.parquet")
    return pd.read_parquet(_PARQUET)


def tasks_root():
    """Per-task workspaces live under repo_root/tasks (historically
    /data/cuda-agent-migration/tasks). Overridable for the same reason."""
    return os.environ.get("MACA_AGENT_TASKS_ROOT",
                          os.path.join(REPO_ROOT, "tasks"))


def total_mem_gb():
    """Total GPU memory in GB, as a float.

    C500 comes in 16.3 GB and 64 GB variants. Older scripts assumed 16.3 and
    sized memory guards from it; on a 64 GB card those guards throttle
    throughput four times harder than the hardware requires. Read the device
    instead of assuming.
    """
    try:
        import torch
        if torch.cuda.is_available():
            _, total = torch.cuda.mem_get_info()
            return total / 1e9
    except Exception:
        pass
    return None
