#!/usr/bin/env python3
"""Local-weights model backend: same interface as call_model, no network.

Purpose: let a fine-tuned local policy (Qwen2.5-Coder-1.5B) drive the agent
loop through the exact same code path as the API model, so a collection /
evaluation run measures the policy itself and nothing else.

Design constraint (important): this file must NOT be wired into
`collect_rollouts.py` by force. That script is data-a1's. It is opt-in via
`--model backend:local:<path>`; the API path stays the default and its
behavior is unchanged.

The returned object mimics the Message shape _stream_one builds: a list of
blocks where a tool call is a dict {"type": "tool_use", "name": ..., "input":
{...}}. extract_attempt() consumes exactly that.

Tool-call parsing: the SFT target format taught by training/sft_smoke.py is

    <tool_use name="write_files"
      kernel_name="..."
      kernel_cu=<json string>
      binding_cpp=<json string>
      model_new_py=<json string>
    </tool_use>

so a local model reproduces that. Parsing is deliberately permissive about
whitespace but strict about JSON validity: if the tool block is malformed we
fall through to text (the turn loop then re-prompts the model), exactly like
the truncated-tool-JSON path in _stream_one.
"""
import json
import os
import re
import time

_LOCAL = None          # cached (model, tokenizer)
_LOCAL_PATH = None     # which weights the cache holds

TOOL_RE = re.compile(
    r'<tool_use\s+name="([^"]+)"(.*?)</tool_use>', re.S)
FIELD_RE = re.compile(r'(\w+)\s*=\s*(json\.dumps\()?')


def _load(path):
    """Load model + tokenizer, cached by path."""
    global _LOCAL, _LOCAL_PATH
    if _LOCAL is not None and _LOCAL_PATH == path:
        return _LOCAL
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    _LOCAL = (model, tok)
    _LOCAL_PATH = path
    return _LOCAL


def _system_to_chat(system):
    """Fold the long system prompt into a chat system message."""
    return system


def _parse_tool_call(text):
    """Pull a <tool_use> block out of generated text, or return None."""
    m = TOOL_RE.search(text)
    if not m:
        return None
    name, body = m.group(1), m.group(2)
    # Each field is  name=<json string>  — the SFT format used json.dumps
    # for the three code fields, so they parse cleanly back.
    fields = {}
    for fm in re.finditer(r'(\w+)\s*=\s*(.*?)(?=\n\s*\w+\s*=|\Z)', body, re.S):
        key, raw = fm.group(1), fm.group(2).strip()
        if key in ("kernel_cu", "binding_cpp", "model_new_py"):
            try:
                fields[key] = json.loads(raw)
            except json.JSONDecodeError:
                # The SFT corpus is cleanly escaped, but a 1.5B policy
                # reproducing ~2 kB of JSON occasionally emits a raw control
                # character (literal newline) inside a string where the
                # corpus had \n. strict=False tolerates exactly that, so a
                # kernel whose only flaw is whitespace escaping is still
                # usable instead of degrading the turn to text.
                try:
                    fields[key] = json.loads(raw, strict=False)
                except json.JSONDecodeError:
                    # Genuinely malformed payload — degrade to text so the
                    # loop re-prompts, same discipline as _stream_one's
                    # truncated-JSON path.
                    return None
        elif key == "kernel_name":
            fields[key] = raw.strip().strip('"')
    required = ("kernel_cu", "binding_cpp", "model_new_py", "kernel_name")
    if not all(fields.get(f) for f in required):
        return None
    return {"type": "tool_use", "name": name, "input": fields}


class _Usage:
    def __init__(self, in_tokens, out_tokens):
        self.input_tokens = in_tokens
        self.output_tokens = out_tokens


class _Block:
    """A content block with attribute access.

    llm_backend's three consumers of resp.content — extract_attempt,
    _content_to_jsonable, _content_to_api — all read block.type /
    block.name / block.input / block.text as ATTRIBUTES (they are written
    against Anthropic SDK objects). A plain dict here silently breaks all
    three: getattr(dict, 'type') is None, so extract_attempt returns None
    and the agent loop spins every turn re-prompting "you did not call
    write_files" even though the model called it. Observed 2026-10-16:
    stop=tool_use on turn 0, yet "did not produce files in 2 turns".
    """
    def __init__(self, kind, **attrs):
        self.type = kind
        self.id = attrs.get("id", "")
        self.name = attrs.get("name", "")
        self.input = attrs.get("input", {})
        self.text = attrs.get("text", "")


class _Message:
    """The shape extract_attempt / _content_to_jsonable expect."""
    def __init__(self, content, stop_reason, usage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


def _flatten_messages(messages):
    """Render the conversation as the flat text collection actually used.

    The recorded `prompt` field (what SFT trained on) is build_user_prompt's
    output: SKILL.md + model.py + feedback, one plain string, no chat
    markers, no SYSTEM_PROMPT. If we wrapped it in apply_chat_template here
    the model would see <|im_start|> tokens it never saw in training and
    silently regress to prose — observed 2026-10-16: same weights, chat
    template -> no tool_use; verbatim flat prompt -> well-formed tool_use.

    Collection rebuilds the FULL prompt every turn (build_user_prompt
    re-reads SKILL.md and model.py and appends the accumulated feedback),
    so `messages` inside one attempt accumulates growing copies. The
    training sample for turn t pairs response_t with that turn's complete
    prompt, not with the concatenated history — so the last user message
    alone is exactly the input distribution the policy was trained on.
    Concatenating the history would duplicate SKILL.md/model.py and drift
    out of distribution.
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        text = "\n".join(
            b.get("text", "") if isinstance(b, dict) else
            getattr(b, "text", "") or "" for b in (content or []))
        if text:
            return text
    return ""


def call_model_local(messages, system, max_tokens=8192, model_path=None,
                     temperature=0.0, flat=True):
    """Generate with the local policy. `model_path` defaults to the smoke
    model written by training/sft_smoke.py.

    `flat=True` feeds the conversation as the plain text collection used
    (matching the SFT input distribution). `flat=False` renders it through
    the tokenizer's chat template instead, for chat-style use.
    """
    import torch

    path = model_path or os.environ.get(
        "LOCAL_MODEL_PATH",
        "/data/maca-agent/training/smoke_model")
    model, tok = _load(path)

    if flat:
        prompt = _flatten_messages(messages)
    else:
        chat = [{"role": "system", "content": _system_to_chat(system)}]
        for m in messages:
            role = m["role"]
            content = m.get("content")
            if isinstance(content, str):
                chat.append({"role": role, "content": content})
                continue
            # API-shaped blocks -> plain text for the local model.
            parts = []
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        parts.append(b["text"])
                    elif b.get("type") == "tool_use":
                        # Echo the prior tool call back as text.
                        parts.append(_render_tool_use(b))
                    elif b.get("type") == "tool_result":
                        parts.append(str(b.get("content", "")))
                else:
                    parts.append(getattr(b, "text", "") or "")
            chat.append({"role": role,
                         "content": "\n".join(p for p in parts if p)})
        prompt = tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = {k: v.to("cuda") for k, v in ids.items()}
    in_tokens = ids["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][in_tokens:], skip_special_tokens=True)
    out_tokens = out.shape[1] - in_tokens
    dt = time.time() - t0

    tool = _parse_tool_call(text)
    if tool is not None:
        content = [_Block("tool_use", id="local_tool", name=tool["name"],
                          input=tool["input"])]
        stop_reason = "tool_use"
    else:
        content = [_Block("text", text=text)]
        stop_reason = "end_turn"

    print(f"    [local] {out_tokens} tok in {dt:.1f}s "
          f"({out_tokens/max(dt,0.001):.1f} tok/s) stop={stop_reason}",
          flush=True)
    return _Message(content, stop_reason, _Usage(in_tokens, out_tokens))


def _render_tool_use(b):
    """Render a tool_use block back as the SFT target text."""
    inp = b.get("input", {})
    return (
        f'<tool_use name="{b.get("name", "write_files")}"\n'
        f'  kernel_name="{inp.get("kernel_name", "")}"\n'
        f'  kernel_cu={json.dumps(inp.get("kernel_cu", ""))}\n'
        f'  binding_cpp={json.dumps(inp.get("binding_cpp", ""))}\n'
        f'  model_new_py={json.dumps(inp.get("model_new_py", ""))}\n'
        f'</tool_use>'
    )


def is_local_spec(model):
    """True for model specs of the form backend:local:<path> or plain paths."""
    if not model:
        return False
    if model.startswith("backend:local:"):
        return True
    # A bare existing directory counts as a local weights path.
    return model.endswith(".safetensors") is False and os.path.isdir(model)


def local_path_from(model):
    s = model[len("backend:local:"):] if model.startswith("backend:local:") \
        else model
    return s or None
