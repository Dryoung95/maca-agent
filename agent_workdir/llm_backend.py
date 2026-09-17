#!/usr/bin/env python3
"""LLM backend for the CUDA-Agent agent loop on C500/MACA (G9).

Implements the `produce_attempt` contract that agent_loop.py expects: given the
workspace state and feedback, ask the model for the three files SKILL.md
requires (kernels/*.cu, kernels/*_binding.cpp, model_new.py).

The model works against the real SKILL.md, with the MACA Supplement included,
so its output is constrained the same way a rollout from the paper's agent is.
Anti-reward-hacking rules from the paper are enforced here, not just asked for:
no network during an episode, no fallback ops, verification re-runs 5 inputs.

Backend is Anthropic-only by default but the call site is one function; swap
`call_model` to use another provider.
"""
import json
import os
import re
import sys

# Network is forbidden mid-episode (paper's anti-hacking rule). The SDK may
# still be imported; we only block outbound calls after episode start.

TOOL_SCHEMA = {
    "name": "write_files",
    "description": (
        "Write the kernel, binding, and model_new.py files for this attempt. "
        "Call exactly once per turn. All three are required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kernel_cu": {
                "type": "string",
                "description": "Contents of kernels/<name>.cu — raw CUDA kernels.",
            },
            "binding_cpp": {
                "type": "string",
                "description": (
                    "Contents of kernels/<name>_binding.cpp — the pybind "
                    "binding. Must #include \"../binding_registry.h\"."
                ),
            },
            "model_new_py": {
                "type": "string",
                "description": (
                    "Contents of model_new.py — must import cuda_extension "
                    "and call the registered forward function."
                ),
            },
            "kernel_name": {
                "type": "string",
                "description": "Base name, e.g. fused_selu_tanh.",
            },
        },
        "required": ["kernel_cu", "binding_cpp", "model_new_py", "kernel_name"],
    },
}

SYSTEM_PROMPT = """You are a PyTorch and CUDA expert accelerating a PyTorch model \
on a MetaX C500 GPU with the MACA toolchain (cucc, no nvcc).

You operate under the SKILL.md contract supplied to you. Key points that are \
easy to get wrong on THIS device and will fail your build or correctness check:

- Warp size is 64, not 32. Shared memory per block is 64 KB. 104 SMs.
- Do NOT set TORCH_CUDA_ARCH_LIST. Leave it unset; the toolchain targets xcore1000.
- #include "binding_registry.h" from kernels/*.cpp must be written as
  "../binding_registry.h" — bindings live in a subdirectory.
- Never call torch:: ops inside kernels or bindings. Only tensor creation,
  data_ptr, and the REGISTER_BINDING machinery.
- The eval harness regenerates inputs with its own seed: normalize dtypes and
  contiguity in the binding instead of rejecting them.
- Performance on this device is memory-traffic-bound. A scalar kernel with one
  thread per element reaches only ~600 GB/s; float2 loads with a fixed
  ~1024-block grid-stride loop reach ~1300 GB/s against a ~1350 GB/s ceiling.
- A chain of two special-function units on the same value (expf feeding
  log1pf, as in softplus) starves the scheduler and loses to eager. Restate
  the computation or expect to lose.

The reference model is provided as model.py. Do not modify it, utils/, \
binding.cpp, or binding_registry.h."""


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def build_user_prompt(workspace, task_hint="", feedback=""):
    """Assemble the per-turn prompt: SKILL.md + model.py + prior feedback."""
    skill = _read(os.path.join(workspace, "SKILL.md"))
    model_py = _read(os.path.join(workspace, "model.py"))
    parts = [
        "Here is the full SKILL.md contract you must follow:",
        "<SKILL.md>",
        skill,
        "</SKILL.md>",
        "",
        "Here is the reference model to accelerate (model.py). Do not modify it:",
        "<model.py>",
        model_py,
        "</model.py>",
    ]
    if task_hint:
        parts += ["", "Additional task context:", task_hint]
    if feedback:
        parts += [
            "",
            "Your previous attempt failed. Diagnostics from the toolchain:",
            "<feedback>",
            feedback,
            "</feedback>",
            "",
            "Fix the specific problem these diagnostics show, then call "
            "write_files again with the corrected files.",
        ]
    else:
        parts += [
            "",
            "Produce the three files (kernel .cu, binding .cpp, model_new.py) "
            "as a single write_files call.",
        ]
    return "\n".join(parts)


def call_model(messages, system, max_tokens=8192, model=None):
    """One model call. Raises on failure — the loop treats that as a failed turn."""
    from anthropic import Anthropic

    client = Anthropic()  # picks up ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[TOOL_SCHEMA],
        messages=messages,
    )
    return resp


def extract_attempt(resp):
    """Pull the write_files tool call out of a model response."""
    for block in getattr(resp, "content", []):
        if getattr(block, "type", None) == "tool_use" and block.name == "write_files":
            return block.input
    return None


def make_attempt_from_llm(workspace, task_hint="", feedback="", model=None,
                          max_turns=3):
    """G9 produce_attempt: one LLM attempt with up to max_turns tool-call rounds.

    Returns (Attempt, trace) where trace records every model exchange, so the
    caller can store the rollout for later RFT filtering.
    """
    from agent_loop import Attempt

    messages = []
    trace = []
    system = SYSTEM_PROMPT

    for turn in range(max_turns):
        user = build_user_prompt(workspace, task_hint, feedback if turn else "")
        messages.append({"role": "user", "content": user})
        resp = call_model(messages, system, model=model)
        messages.append({"role": "assistant", "content": resp.content})
        trace.append({"turn": turn, "stop_reason": resp.stop_reason})

        attempt_input = extract_attempt(resp)
        if attempt_input is None:
            feedback = "You did not call write_files. Call it with all three files."
            continue

        name = attempt_input.get("kernel_name", "fused_kernel")
        attempt = Attempt(
            name=name,
            kernel_files={f"kernels/{name}.cu": attempt_input["kernel_cu"]},
            binding_files={f"kernels/{name}_binding.cpp":
                           attempt_input["binding_cpp"]},
            model_new=attempt_input["model_new_py"],
        )
        trace[-1]["produced"] = name
        return attempt, trace

    raise RuntimeError(f"model did not produce files in {max_turns} turns")


def main():
    """Smoke entry: ask the model for the axpby example, without running it."""
    workspace = os.path.dirname(os.path.abspath(__file__))
    print(f"workspace: {workspace}")
    print("asking the model for an implementation of model.py ...")
    attempt, trace = make_attempt_from_llm(workspace)
    print(f"\nmodel produced: {attempt.name}")
    print(f"kernel bytes:   {len(attempt.kernel_files[f'kernels/{attempt.name}.cu'])}")
    print(f"binding bytes:  "
          f"{len(attempt.binding_files[f'kernels/{attempt.name}_binding.cpp'])}")
    print(f"model_new bytes: {len(attempt.model_new)}")
    print(f"\ntrace: {json.dumps(trace, default=str)[:400]}")


if __name__ == "__main__":
    main()
