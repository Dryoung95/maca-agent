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
  log1pf, as in softplus) starves the scheduler: a fused `expf->log1pf`
  kernel measures ~0.5x. The fix is the algebraic restatement
  `log1p(exp(y)) = y + log1p(e^-y)`, which breaks the dependency and is a
  large real gain — but measured under the official KernelBench protocol
  (reference = eager) the restated kernel still ran 0.77x on this device.
  Treat chained-SFU tasks as stretch targets, not sweet spots: try the
  restate if you have budget, but prefer a memory-bound fusion where the
  traffic savings are real.

The reference model is provided as model.py. Do not modify it, utils/, \
binding.cpp, or binding_registry.h.
- `REGISTER_BINDING(name, func)` expands to a static declaration, so the call
  MUST end with a semicolon: `REGISTER_BINDING(k, register_k);`. Omitting it
  fails the build with a confusing `expected ',' or ';' at end of input`
  pointed at `binding_registry.h` — the real cause is your missing semicolon,
  not the header.

Keep your text analysis SHORT — a few sentences: the op shape, the memory \
traffic to remove, the vectorization choice. Put the actual kernel, binding, \
and model_new code in the write_files call, never in the text. Writing the \
kernel out in prose runs you out of tokens before you can call the tool, \
which produces no files and wastes the turn."""


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

    Three things that had to be right simultaneously, each of which separately
    looked like a different bug:

    **thinking must be disabled.** This endpoint enables extended thinking by
    default, and with it on the model burns the ENTIRE output budget on
    hidden reasoning: the stream contains no content_block_* events at all,
    yet output_tokens runs to the max and stop_reason=max_tokens. No
    write_files call is ever emitted. That is what actually broke row 3579 —
    not the token limit. 8192 truncates, 16384 truncates, and 16384 over the
    non-streaming path dies at the 45-minute gateway cutoff first. With
    thinking disabled the same task completes in 2623 output tokens.

    max_tokens stays 16384 even though real usage is ~3k: the model writes a
    text analysis before the tool call, and that preamble varies enough that
    8192 intermittently truncates it (observed on row 3579 attempt 2 turn 0).
    The headroom is cheap because a 3k-token stream finishes in ~2 min; the
    limit only matters when it is actually reached.

    **streaming, not messages.create.** Two independent reasons. (1) Speed:
    the non-streaming path makes the gateway buffer the whole generation, and
    throughput drops to ~7-16 tok/s; streaming runs ~22-97 tok/s, a 5x
    difference. (2) The gateway cuts any connection — streaming or not — at
    a hard ~600s, so any non-streaming call long enough to matter is
    uncompletable. A 16k-token stream survives only because streaming makes
    it fast enough to finish first; the real fix was disabling thinking.

    **retry transient transport faults.** A 502 from the proxy used to cost
    the whole episode, since run_g9 treats a raised exception as a failed
    attempt. 5xx/429/conn/timeout retry up to 3 times (10/30/90s backoff);
    other 4xx raise immediately.
    """
    from anthropic import Anthropic
    from anthropic import (APIConnectionError, APIStatusError,
                           APITimeoutError)

    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Local-weights dispatch: `backend:local:<path>` (or an existing
    # directory) runs a fine-tuned policy through the same call site, so a
    # collection/eval run measures the policy and nothing else. The API
    # path below is unchanged in behavior.
    import local_backend
    if local_backend.is_local_spec(model):
        path = local_backend.local_path_from(model)
        return local_backend.call_model_local(
            messages, system, max_tokens=max_tokens, model_path=path)

    client = Anthropic(timeout=1800.0)

    RETRIES = 3
    wait = [10, 30, 90]
    last = None
    for attempt in range(RETRIES):
        try:
            return _stream_one(client, model, messages, system, max_tokens)
        except APIConnectionError as e:
            last = e          # proxy refused / dropped: request never landed
        except APITimeoutError as e:
            last = e
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


class _TextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _ToolBlock:
    def __init__(self, id_, name, input_):
        self.type = "tool_use"
        self.id = id_
        self.name = name
        self.input = input_


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Message:
    """The shape extract_attempt / _content_to_jsonable / run_g9 expect."""
    def __init__(self, content, stop_reason, usage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


def _stream_one(client, model, messages, system, max_tokens):
    """One streaming request; content accumulated from SSE events.

    get_final_message() is NOT used: on this gateway it can come back with an
    empty content list (the message_delta/message_stop ordering is off), so
    the blocks are rebuilt from the events themselves.
    """
    blocks = {}          # index -> {"type","id","name", text/json accumulator}
    order = []           # keep insertion order for the final content list
    stop_reason = None
    out_tokens = in_tokens = 0

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "disabled"},
        tools=[TOOL_SCHEMA],
        messages=messages,
    ) as stream:
        for event in stream:
            k = event.type
            if k == "message_start":
                in_tokens = event.message.usage.input_tokens
            elif k == "content_block_start":
                b = event.content_block
                idx = event.index
                blocks[idx] = {"type": b.type,
                               "id": getattr(b, "id", ""),
                               "name": getattr(b, "name", ""),
                               "buf": ""}
                order.append(idx)
            elif k == "content_block_delta":
                d = event.delta
                dt = getattr(d, "type", "?")
                cell = blocks.get(event.index)
                if cell is None:
                    continue
                if dt == "text_delta":
                    cell["buf"] += d.text
                elif dt == "input_json_delta":
                    cell["buf"] += d.partial_json
            elif k == "message_delta":
                if getattr(event.usage, "output_tokens", None):
                    out_tokens = event.usage.output_tokens
                if event.delta.stop_reason:
                    stop_reason = event.delta.stop_reason

    content = []
    for idx in order:
        cell = blocks[idx]
        if cell["type"] == "text":
            content.append(_TextBlock(cell["buf"]))
        elif cell["type"] == "tool_use":
            try:
                payload = json.loads(cell["buf"]) if cell["buf"] else {}
            except json.JSONDecodeError:
                # Truncated mid-JSON. Do NOT fabricate a tool_use with a
                # partial payload: extract_attempt would accept it and the
                # caller would KeyError on the missing fields. Report it as
                # text so the turn loop treats this as "no write_files call"
                # and asks the model again.
                content.append(_TextBlock(
                    f"[truncated tool_use for {cell['name']}, "
                    f"{len(cell['buf'])} chars of JSON arrived]"))
                continue
            content.append(_ToolBlock(cell["id"], cell["name"], payload))

    return _Message(content, stop_reason, _Usage(in_tokens, out_tokens))


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


def _content_to_api(content):
    """Content blocks -> the assistant-message shape the API accepts on input.

    The API's assistant role takes content as a list of block dicts. Tool
    blocks need their id back so the next turn can be paired with a
    tool_result if one is ever needed. This is a separate function from
    _content_to_jsonable because the two consumers differ: the trace is for
    persistence, this is for the request body.
    """
    out = []
    for block in content or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            out.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            out.append({"type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input})
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
        # The cross-attempt feedback (compile stderr / verification diff /
        # performance gap from run_g9) arrives via `feedback` and MUST be
        # passed on turn 0. The previous form, `feedback if turn else ""`,
        # suppressed it on the first turn — and since most attempts produce
        # files on turn 0 and return immediately, the model never saw why
        # its last attempt failed. Observed on row 1582: three attempts,
        # byte-identical 21811-char prompts, the same undeclared-identifier
        # error each time.
        user = build_user_prompt(workspace, task_hint, feedback)
        messages.append({"role": "user", "content": user})
        resp = call_model(messages, system, model=model)
        assistant_jsonable = _content_to_jsonable(resp.content)
        # Feed the next turn an API-shaped assistant message. The API accepts
        # a plain list of content-block dicts for the assistant role, which is
        # what _content_to_jsonable already produces.
        messages.append({"role": "assistant",
                         "content": _content_to_api(resp.content)})
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

        # A write_files call missing a required field is a recoverable turn,
        # not a dead attempt: run_g9 would KeyError here and break the whole
        # episode, discarding the remaining attempts. Ask the model to
        # complete the call instead.
        missing = [f for f in ("kernel_cu", "binding_cpp", "model_new_py")
                   if not attempt_input.get(f)]
        if missing:
            feedback = (
                "Your write_files call is missing required field(s): "
                f"{', '.join(missing)}. Re-call write_files with all four "
                "fields (kernel_cu, binding_cpp, model_new_py, kernel_name)."
            )
            trace[-1]["note"] = f"missing fields: {missing}"
            _dump_trace(workspace, trace, tag=f"missing_turn{turn}")
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
