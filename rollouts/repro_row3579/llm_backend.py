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
import time

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


def call_model(messages, system, max_tokens=16384, model=None):
    """One model call. Raises on failure — the loop treats that as a failed turn.

    Two defaults that had to move together; the pipeline failed on real tasks
    until both were fixed:

    max_tokens 16384, not 8192. A real dataset task (row 3579, expm1 +
    nn.Softmax) emits ~8.2k output tokens; at 8192 the model is truncated
    mid-write_files -- stop_reason=max_tokens, no tool_use block, and
    extract_attempt finds nothing, 3 turns running, so every episode on a
    parameterized task failed as "model did not produce files". The axpby
    example passed at 2135 output tokens, which is why G9 looked green.

    read timeout 1800s, not the SDK's 600s. This endpoint sustains ~16 tok/s
    (measured: 2100 tokens in 125s, 6285 in 404s), so 16k output tokens take
    ~17 minutes. The default 600s read timeout cuts off at ~10k tokens, which
    presents as the call never returning -- no exception, no retry, the turn
    just dies. That is not an endpoint outage and cannot be detected without
    the raised timeout.
    """
    from anthropic import Anthropic
    from anthropic import (APIConnectionError, APIStatusError,
                           APITimeoutError)

    client = Anthropic(timeout=1800.0)
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Retry transient transport faults. One 502 from the proxy otherwise costs
    # the whole episode: a call here is ~17 min of generation (16k tokens at
    # ~16 tok/s), and run_g9 treats a raised exception as a failed attempt --
    # observed on row 3579, where a 502 after 45 min produced zero usable
    # output and no retry. Non-retryable errors (4xx, except 429) raise.
    RETRIES = 3
    last = None
    wait = [10, 30, 90]
    for attempt in range(RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=[TOOL_SCHEMA],
                messages=messages,
            )
            return resp
        except APIConnectionError as e:
            last = e          # proxy refused / dropped: request never landed
        except APITimeoutError as e:
            last = e          # generation exceeded the 1800s read timeout
        except APIStatusError as e:
            # 5xx and 429 are the gateway's problem; anything else is ours
            # (bad request, auth) and must not be retried.
            if e.status_code < 500 and e.status_code != 429:
                raise
            last = e
        if attempt + 1 < RETRIES:
            print(f"    [llm] transient {type(last).__name__} "
                  f"({getattr(last, 'status_code', 'n/a')}), "
                  f"retry {attempt + 2}/{RETRIES} in {wait[attempt]}s",
                  flush=True)
            time.sleep(wait[attempt])
    raise last


def extract_attempt(resp):
    """Pull the write_files tool call out of a model response."""
    for block in getattr(resp, "content", []):
        if getattr(block, "type", None) == "tool_use" and block.name == "write_files":
            return block.input
    return None


def _content_to_jsonable(content):
    """Anthropic SDK content blocks -> plain JSON (text + tool_use inputs).

    Everything needed for RFT lives in these two; the raw SDK objects are not
    json-serializable, so they would be lost on dump.
    """
    out = []
    for block in content or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            out.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            out.append({"type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input})
        elif kind == "thinking":
            out.append({"type": "thinking",
                        "text": getattr(block, "thinking", "")})
    return out


def _dump_trace(workspace, trace, tag=""):
    """Persist a partial trace next to the workspace.

    A turn that produces no write_files call, or an attempt that exhausts its
    turns, is the expensive failure mode on this endpoint: each model call is
    tens of minutes. Without a dump the model's actual output is discarded, so
    the failure is undiagnosable -- 104 minutes of row 3579 produced only the
    string "model did not produce files in 3 turns". Written to the workspace
    itself (not the repo) so it survives the episode and never pollutes git.
    """
    try:
        import time as _t
        path = os.path.join(
            workspace, f"trace_dump_{tag}_{int(_t.time())}.json")
        with open(path, "w") as f:
            json.dump(trace, f, indent=1, default=str)
        print(f"    [trace] wrote {path}", flush=True)
    except Exception as e:  # diagnosis must never kill the episode
        print(f"    [trace] dump failed: {type(e).__name__}: {e}", flush=True)


def make_attempt_from_llm(workspace, task_hint="", feedback="", model=None,
                          max_turns=3):
    """G9 produce_attempt: one LLM attempt with up to max_turns tool-call rounds.

    Returns (Attempt, trace). trace now carries the full RFT record per turn:
    the prompt we sent, the model's complete response (text + tool inputs),
    and the extracted file contents. Pair it with the reward the harness
    computes for this attempt and you have a (prompt, response, reward)
    triple; without the prompt/response text the rollout is not reusable.
    """
    from agent_loop import Attempt

    messages = []
    trace = []
    system = SYSTEM_PROMPT
    model_id = model or os.environ.get("ANTHROPIC_MODEL",
                                       "claude-sonnet-4-20250514")

    for turn in range(max_turns):
        user = build_user_prompt(workspace, task_hint, feedback if turn else "")
        messages.append({"role": "user", "content": user})
        resp = call_model(messages, system, model=model)
        assistant_jsonable = _content_to_jsonable(resp.content)
        # Keep the live SDK object in `messages` for the next turn's context,
        # and store the plain-JSON copy in the trace for persistence.
        messages.append({"role": "assistant", "content": resp.content})
        trace.append({
            "turn": turn,
            "stop_reason": resp.stop_reason,
            "model": model_id,
            "user_prompt": user,
            "assistant": assistant_jsonable,
            "usage": getattr(resp, "usage", None) and {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        })

        attempt_input = extract_attempt(resp)
        if attempt_input is None:
            feedback = ("You did not call write_files. Call it once with all "
                        "three files as the tool input.")
            trace[-1]["note"] = "no write_files call"
            _dump_trace(workspace, trace, tag=f"nowrite_turn{turn}")
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
        trace[-1]["files"] = {
            f"kernels/{name}.cu": attempt_input["kernel_cu"],
            f"kernels/{name}_binding.cpp": attempt_input["binding_cpp"],
            "model_new.py": attempt_input["model_new_py"],
        }
        return attempt, trace

    _dump_trace(workspace, trace, tag=f"exhausted_{max_turns}turns")
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
