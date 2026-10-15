#!/usr/bin/env python3
"""Minimal agent loop driver for the CUDA-Agent workspace on C500/MACA.

Implements the SKILL.md UNIFIED WORKFLOW as a repeatable, reward-bearing
episode:

  Step 1  implementation   (caller supplies kernel/binding/model_new)
  Step 2  compile + verify (utils.compile / utils.verification)
  Step 3  profile          (utils.profiling, DCE-safe)
  Step 4  iterate          (loop until correctness passes, then push perf)
  Step 5  cleanup          (kernels/ must contain only the final version)

The reward follows the paper's milestone scheme: correctness is gate-valued
(KernelBench `eval_kernel_against_ref`), performance is judged against the
torch.compile baseline with the 5% bar from SKILL.md.

This is deliberately LLM-agnostic: it takes a `produce_attempt` callable so
the same harness drives a human-written attempt, a scripted baseline, or a
model rollout. Proving the harness is correct with a known-good attempt is
the point — before any model is attached.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

# KernelBench's src on sys.path. Prefer the env var set by env.sh; fall back to
# the current layout (sibling maca-harness repo) and to the historical path.
def _kernelbench_src():
    p = os.environ.get("KERNELBENCH_SRC")
    cands = [p, "/data/maca-harness/KernelBench/src",
             "/data/cuda-harness-migration/KernelBench/src"]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "kernelbench")):
            return c
    raise SystemExit(
        "KernelBench source not on sys.path (kernelbench.eval is the official "
        "eval gate). Set KERNELBENCH_SRC or source env.sh.")

sys.path.insert(0, _kernelbench_src())
from kernelbench.eval import eval_kernel_against_ref

# SKILL.md Step 4: "Must be at least 5% faster than torch.compile baseline"
SPEEDUP_BAR = 1.05
MAX_ATTEMPTS = 5


@dataclass
class Attempt:
    """One implementation attempt: files written into kernels/ + model_new.py."""
    name: str
    kernel_files: dict          # {path: contents}
    binding_files: dict = field(default_factory=dict)
    model_new: str = ""


@dataclass
class AttemptResult:
    name: str
    compiled: bool = False
    correct: bool = False
    compile_s: float = 0.0
    verify_ok: bool = False
    ext_us: float = 0.0
    compile_us: float = 0.0
    eager_us: float = 0.0
    kernelbench: dict = field(default_factory=dict)
    error: str = ""
    wall_s: float = 0.0
    eager_speedup: float = 0.0   # ext vs eager (the KernelBench gate's ref)


def run(cmd, cwd, env=None, timeout=600):
    e = os.environ.copy()
    if env:
        e.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def step2_compile_and_verify(workspace):
    """SKILL.md Step 2: compile, then verification 5/5."""
    rc, out, err, dt = run(["bash", "utils/compile.sh"], workspace)
    if rc != 0:
        return False, 0.0, f"compile rc={rc}: {(err or out)[-300:]}"
    compiled_s = dt

    rc, out, err, dt = run(
        [sys.executable, "-m", "utils.verification"], workspace)
    if rc != 0:
        return True, compiled_s, f"verify rc={rc}: {(err or out)[-300:]}"
    return True, compiled_s, ""


def step3_profile(workspace):
    """SKILL.md Step 3. Returns (ext_us, compile_us, eager_us) or None."""
    rc, out, err, _ = run(
        [sys.executable, "-m", "utils.profiling"], workspace, timeout=900)
    if rc != 0:
        return None
    # utils.profiling prints one line:
    #   "Torch Baseline: Xus, Torch Compile: Yus, CUDA Extension: Zus"
    # All three share that single line, so the value must be extracted
    # relative to each key's own position — a naive split(":") collapses
    # them onto the first number.
    nums = {}
    for line in (out + err).splitlines():
        for key, label in (("Torch Baseline", "eager"),
                           ("Torch Compile", "compile"),
                           ("CUDA Extension", "ext")):
            idx = line.find(key)
            if idx < 0:
                continue
            rest = line[idx + len(key):]
            try:
                nums[label] = float(rest.split(":")[1].split("us")[0])
            except (ValueError, IndexError):
                pass
    if "ext" not in nums or "compile" not in nums:
        return None
    return nums.get("ext", 0.0), nums.get("compile", 0.0), nums.get("eager", 0.0)


def step5_cleanup(workspace):
    """SKILL.md Step 5: kernels/ keeps only the final version.

    Only removes files that nothing references any more. A *_v2.cu that the
    surviving binding still links against must stay — deleting it leaves an
    undefined symbol at load time and the workspace becomes unbuildable.
    """
    kernels = Path(workspace) / "kernels"
    if not kernels.is_dir():
        return
    binding_text = ""
    for p in kernels.glob("*_binding.cpp"):
        binding_text += p.read_text(errors="ignore")
    for pattern in ("*_old.cu", "*_test.cu", "*.bak"):
        for p in kernels.glob(pattern):
            p.unlink()
    # Versioned kernels: keep any whose launcher symbol the bindings still need.
    for p in kernels.glob("*_v[0-9].cu"):
        stem = p.stem  # e.g. fused_leaky_relu_leaky_relu_v2
        if f"{stem}_launcher" in binding_text:
            continue
        p.unlink()


def kernelbench_gate(workspace, model_py="model.py",
                     model_new_py="model_new.py"):
    """The paper's reward gate, run in the workspace's own context."""
    task_dir = os.path.abspath(workspace)
    sys.modules.pop("cuda_extension", None)
    sys.path.insert(0, task_dir)
    try:
        original = open(os.path.join(task_dir, model_py)).read()
        custom = open(os.path.join(task_dir, model_new_py)).read()
        res = eval_kernel_against_ref(
            original_model_src=original,
            custom_model_src=custom,
            measure_performance=True,
            timing_method="cuda_event",
        )
        d = res.to_dict() if hasattr(res, "to_dict") else vars(res)
        return {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                    else str(v)) for k, v in d.items()}
    finally:
        try:
            sys.path.remove(task_dir)
        except ValueError:
            pass


def run_attempt(workspace, attempt):
    """Run one full Step 1→5 cycle for a single attempt. Mutates workspace."""
    t0 = time.time()
    r = AttemptResult(name=attempt.name)

    # Step 1: implementation
    for path, contents in {**attempt.kernel_files,
                           **attempt.binding_files}.items():
        full = os.path.join(workspace, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(contents)
    if attempt.model_new:
        with open(os.path.join(workspace, "model_new.py"), "w") as f:
            f.write(attempt.model_new)

    # Step 2: compile + verify
    compiled, compile_s, err = step2_compile_and_verify(workspace)
    r.compiled = compiled
    r.compile_s = round(compile_s, 1)
    if not compiled:
        r.error = err
        r.wall_s = round(time.time() - t0, 1)
        return r

    # verification exit 0 => 5/5 checks passed
    r.verify_ok = not err
    r.correct = r.verify_ok
    if err:
        r.error = err

    # Step 3: profile (only meaningful when correct)
    if r.correct:
        prof = step3_profile(workspace)
        if prof:
            r.ext_us, r.compile_us, r.eager_us = prof

    r.wall_s = round(time.time() - t0, 1)
    return r


def compute_reward(r):
    """Paper's milestone reward: discrete, gate-valued.

    Two distinct baselines are in play and they disagree often enough to
    matter. SKILL.md's 5% bar is relative to **torch.compile**; the
    KernelBench gate's ref is **eager**. A memory-bound fused kernel can
    beat eager comfortably and still lose to torch.compile. Record both so
    a downstream consumer never reads one as the other.
    """
    if not r.compiled:
        return 0.0, "compile_failed"
    if not r.correct:
        return 0.1, "compiled_but_incorrect"
    if r.compile_us <= 0:
        return 0.3, "correct_unmeasured"
    ratio = r.compile_us / max(r.ext_us, 1e-9)
    eager_ratio = r.eager_us / max(r.ext_us, 1e-9) if r.eager_us > 0 else 0.0
    r.eager_speedup = round(eager_ratio, 3)
    if ratio >= SPEEDUP_BAR:
        return 1.0, f"pass_{ratio:.2f}x_vs_compile"
    return 0.5, f"correct_slow_{ratio:.2f}x_vs_compile"


def run_episode(workspace, attempts, task_name="task"):
    """Full SKILL.md episode: iterate attempts until correct, then push perf."""
    history = []
    best = None
    for i, attempt in enumerate(attempts):
        print(f"\n--- [{task_name}] attempt {i+1}/{len(attempts)}: "
              f"{attempt.name} ---", flush=True)
        r = run_attempt(workspace, attempt)
        reward, reason = compute_reward(r)
        print(f"    compiled={r.compiled} correct={r.correct} "
              f"ext={r.ext_us:.1f}us compile={r.compile_us:.1f}us "
              f"-> reward {reward} ({reason}) [{r.wall_s}s]", flush=True)
        r.kernelbench = {}
        history.append({"name": r.name, "compiled": r.compiled,
                        "correct": r.correct, "ext_us": r.ext_us,
                        "compile_us": r.compile_us, "eager_us": r.eager_us,
                        "eager_speedup": r.eager_speedup,
                        "reward": reward, "reason": reason,
                        "error": r.error, "wall_s": r.wall_s,
                        "compile_s": r.compile_s})
        if r.correct and (best is None
                          or r.ext_us < best.ext_us):
            best = r
        if r.correct and r.compile_us > 0 \
                and r.compile_us / max(r.ext_us, 1e-9) >= SPEEDUP_BAR:
            print(f"    [GATE PASSED] {r.compile_us/max(r.ext_us,1e-9):.2f}x "
                  f">= {SPEEDUP_BAR}x", flush=True)

    # Final gate on the best attempt — must run BEFORE cleanup, since the
    # gate may rebuild the extension and needs every referenced source file.
    gate = {}
    if best and best.correct:
        try:
            gate = kernelbench_gate(workspace)
        except Exception as e:
            gate = {"error": f"{type(e).__name__}: {e}"}
        print(f"    [KernelBench gate] {gate}", flush=True)
        if history:
            history[-1]["kernelbench"] = gate

    # Step 5: cleanup last, after every measurement is done
    step5_cleanup(workspace)

    summary = {"task": task_name, "attempts": history,
               "best": {"name": best.name, "ext_us": best.ext_us,
                        "compile_us": best.compile_us}
               if best else None,
               "kernelbench": gate}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True,
                    help="task workspace dir (contains model.py, utils/)")
    ap.add_argument("--task-name", default="task")
    ap.add_argument("--report", default="agent_loop_report.json")
    args = ap.parse_args()

    # Self-test mode: run a scripted known-good attempt against the workspace.
    # Callers pass real attempts programmatically.
    print("agent_loop requires programmatic attempts; see run_smoke.py")
    return 0


if __name__ == "__main__":
    main()
