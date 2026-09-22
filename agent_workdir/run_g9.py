#!/usr/bin/env python3
"""G9: LLM-driven episode through the agent loop.

Wires llm_backend into agent_loop.run_episode. The model reads SKILL.md and
model.py, writes its three files, and the harness compiles / verifies /
profiles / gates it exactly as a training rollout would.

Multi-turn debugging is handled by the harness: if verification or compile
fails, the stderr and verification diff are fed back and the model retries,
up to max_attempts.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_loop import Attempt, run_attempt, compute_reward, kernelbench_gate
from llm_backend import make_attempt_from_llm

MAX_ATTEMPTS = 3


def gather_feedback(result):
    """Turn a failed or underperforming AttemptResult into model-readable
    diagnostics. SKILL.md Step 4 requires iterating for performance too, not
    just correctness — a correct-but-slow kernel is not done."""
    if not result.compiled:
        return f"Compilation failed:\n{result.error}"
    if not result.correct:
        return (
            f"Compiled successfully, but correctness verification failed:\n"
            f"{result.error}\n\nCheck boundary conditions (tid < size), "
            f"indexing, dtype handling in the binding, and whether the "
            f"kernel math matches model.py exactly."
        )
    # Correct but below the 5% bar: push for the optimizations SKILL.md lists
    ratio = result.compile_us / max(result.ext_us, 1e-9)
    return (
        f"Correctness passed, but performance is below the required bar: "
        f"your kernel {result.ext_us:.1f}us vs torch.compile "
        f"{result.compile_us:.1f}us = {ratio:.2f}x. The minimum is 1.05x.\n\n"
        f"On this device the win comes from removing memory traffic. Apply "
        f"vectorized loads (float2/float4), grid-stride loops with a fixed "
        f"block count instead of one thread per element, and re-check that "
        f"you are not launching a grid far finer than the 104 SMs can fill. "
        f"Target ~1300 GB/s against a ~1350 GB/s copy ceiling."
    )


def run_llm_episode(workspace, task_name="g9", max_attempts=MAX_ATTEMPTS,
                    model=None, task_hint="", do_gate=True):
    history = []
    feedback = ""
    best = None

    for i in range(max_attempts):
        print(f"\n--- [{task_name}] LLM attempt {i+1}/{max_attempts} ---",
              flush=True)
        try:
            attempt, trace = make_attempt_from_llm(
                workspace, task_hint=task_hint, feedback=feedback, model=model)
        except Exception as e:
            print(f"    model call failed: {type(e).__name__}: {e}", flush=True)
            history.append({"attempt": f"model_call_{i}",
                            "error": f"{type(e).__name__}: {e}"})
            break

        r = run_attempt(workspace, attempt)
        reward, reason = compute_reward(r)
        print(f"    compiled={r.compiled} correct={r.correct} "
              f"ext={r.ext_us:.1f}us compile={r.compile_us:.1f}us "
              f"-> reward {reward} ({reason}) [{r.wall_s}s]", flush=True)

        entry = {"attempt": r.name, "compiled": r.compiled,
                 "correct": r.correct, "ext_us": r.ext_us,
                 "compile_us": r.compile_us, "eager_us": r.eager_us,
                 "eager_speedup": r.eager_speedup,
                 "reward": reward, "reason": reason,
                 "error": r.error, "wall_s": r.wall_s,
                 "trace": trace}
        history.append(entry)

        if r.correct and (best is None or r.ext_us < best.ext_us):
            best = r
        if r.correct and r.compile_us > 0 \
                and r.compile_us / max(r.ext_us, 1e-9) >= 1.05:
            print(f"    [GATE PASSED] {r.compile_us/max(r.ext_us,1e-9):.2f}x",
                  flush=True)
            break
        feedback = gather_feedback(r)
        if r.correct:
            feedback += (
                "\n\nYour previous kernel was correct — keep its math intact "
                "and change only the performance characteristics. Do not "
                "rewrite the computation."
            )

    gate = {}
    if best and best.correct and do_gate:
        try:
            gate = kernelbench_gate(workspace)
            print(f"    [KernelBench gate] compiled={gate.get('compiled')} "
                  f"correctness={gate.get('correctness')} "
                  f"runtime={gate.get('runtime')}ms "
                  f"ref={gate.get('ref_runtime')}ms", flush=True)
        except Exception as e:
            gate = {"error": f"{type(e).__name__}: {e}"}
        if history:
            history[-1]["kernelbench"] = gate

    return {"task": task_name, "history": history,
            "best": {"name": best.name, "ext_us": best.ext_us,
                     "compile_us": best.compile_us} if best else None,
            "kernelbench": gate}


def to_training_samples(summary):
    """Flatten an episode summary into (prompt, response, reward) records.

    One record per turn that produced files. The response is the model's
    write_files call as given; the reward is the harness verdict for that
    attempt. Downstream RFT/SFT consumes this list directly — the episode
    report alone is not usable for training because its `trace` is nested by
    attempt, not one row per supervised example.
    """
    samples = []
    for h in summary.get("history", []):
        trace = h.get("trace") or []
        for turn in trace:
            if "files" not in turn:
                continue
            assistant = next((b for b in turn.get("assistant", [])
                              if b.get("type") == "tool_use"
                              and b.get("name") == "write_files"), None)
            if assistant is None:
                continue
            samples.append({
                "task": summary.get("task"),
                "attempt": h.get("attempt"),
                "turn": turn.get("turn"),
                "model": turn.get("model"),
                "prompt": turn.get("user_prompt"),
                "response": {"type": "tool_use",
                             "name": "write_files",
                             "input": assistant.get("input", {})},
                "reward": h.get("reward"),
                "reason": h.get("reason"),
                "compiled": h.get("compiled"),
                "correct": h.get("correct"),
                "ext_us": h.get("ext_us"),
                "compile_us": h.get("compile_us"),
                "eager_us": h.get("eager_us"),
            })
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=os.path.dirname(
        os.path.abspath(__file__)))
    ap.add_argument("--task-name", default="g9_axpby")
    ap.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    ap.add_argument("--model", default=None)
    ap.add_argument("--report", default="g9_report.json")
    ap.add_argument("--samples", default=None,
                    help="also write (prompt, response, reward) rows here")
    args = ap.parse_args()

    summary = run_llm_episode(args.workspace, task_name=args.task_name,
                              max_attempts=args.max_attempts,
                              model=args.model)
    ok = bool(summary.get("kernelbench", {}).get("correctness"))
    print(f"\n=== G9 result: "
          f"{'EPISODE PASS' if ok else 'EPISODE FAIL'} ===")
    for h in summary["history"]:
        print(f"  {h.get('attempt','?')}: compiled={h.get('compiled')} "
              f"correct={h.get('correct')} reward={h.get('reward')} "
              f"({h.get('reason','')})")

    samples = to_training_samples(summary)
    path = args.samples or os.path.join(
        os.path.dirname(os.path.abspath(args.report)), "rollout_samples.jsonl")
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    print(f"rollout samples: {len(samples)} row(s) -> {path}")
    positive = [s for s in samples if (s.get("reward") or 0) >= 0.5]
    print(f"  positive (reward>=0.5): {len(positive)}")

    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
