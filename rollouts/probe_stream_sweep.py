#!/usr/bin/env python3
"""流式 max_tokens sweep：找到 row 3579 完整响应真正需要的上限。

结论链（都写在日志里，避免再走弯路）：
  8192  非流式：截断，无 tool_use（已确认）
  16384 非流式：45 min 后网关 502，从未完成（run2/3/repro_one 共 4 次）
  16384 流式  ：4.2 min 完成，stop=max_tokens，content blocks 为空 —— 仍然截断

流式把单次调用从 ~45 min 压到 ~4 min（64.8 tok/s vs 非流式 ~7-16），
所以 sweep 现在可行。本探针逐级上调 max_tokens，直到 stop_reason=end_turn
且 write_files 工具调用完整产出，记录每一级的实际输出量。

content 累积自己从事件里攒：这份网关下 get_final_message() 返回空 content
（message_stop 之后还收到 message_delta，顺序反常），所以不能依赖它。
"""
import json
import os
import sys
import time

REPO = "/data/maca-agent"
sys.path.insert(0, os.path.join(REPO, "agent_workdir"))
os.chdir(REPO)

from llm_backend import SYSTEM_PROMPT  # noqa: E402

TOOL = {
    "name": "write_files",
    "description": ("Write the kernel, binding, and model_new.py files for "
                    "this attempt. Call exactly once per turn. All three "
                    "are required."),
    "input_schema": {
        "type": "object",
        "properties": {
            "kernel_cu": {"type": "string"},
            "binding_cpp": {"type": "string"},
            "model_new_py": {"type": "string"},
            "kernel_name": {"type": "string"},
        },
        "required": ["kernel_cu", "binding_cpp", "model_new_py", "kernel_name"],
    },
}


def one_call(client, model, user, max_tokens):
    """一次流式调用，自己累积文本与工具输入。返回诊断 dict。"""
    t0 = time.time()
    text_parts = []
    tool_partial = {}
    tool_blocks = []       # (id, name) 来自 content_block_start
    stop = None
    out_tokens = in_tokens = None
    first_event = None
    first_delta = last_delta = None
    n_deltas = 0
    delta_log = []         # (elapsed_s, cumulative_json_chars) 抽样

    try:
        with client.messages.stream(
            model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
            tools=[TOOL], messages=[{"role": "user", "content": user}],
        ) as stream:
            for ev in stream:
                now = time.time()
                if first_event is None:
                    first_event = now - t0
                k = ev.type
                if k == "message_start":
                    in_tokens = ev.message.usage.input_tokens
                elif k == "content_block_start":
                    b = ev.content_block
                    tool_blocks.append((getattr(b, "id", "?"), b.type,
                                        getattr(b, "name", "-")))
                elif k == "content_block_delta":
                    d = ev.delta
                    dt = getattr(d, "type", "?")
                    if first_delta is None:
                        first_delta = now - t0
                    last_delta = now - t0
                    n_deltas += 1
                    if dt == "text_delta":
                        text_parts.append(d.text)
                    elif dt == "input_json_delta":
                        tool_partial["_json"] = \
                            tool_partial.get("_json", "") + d.partial_json
                    # 每 100 个 delta 记一个点，出错时能看到流量是否在持续
                    if n_deltas % 100 == 0:
                        delta_log.append((round(now - t0, 1),
                                          len(tool_partial.get("_json", ""))))
                elif k == "message_delta":
                    if getattr(ev.usage, "output_tokens", None):
                        out_tokens = ev.usage.output_tokens
                    if ev.delta.stop_reason:
                        stop = ev.delta.stop_reason
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}",
                "elapsed_s": round(time.time() - t0, 1),
                "first_event_s": round(first_event, 1) if first_event else None,
                "first_delta_s": round(first_delta, 1) if first_delta else None,
                "last_delta_s": round(last_delta, 1) if last_delta else None,
                "n_deltas": n_deltas,
                "tool_json_chars": len(tool_partial.get("_json", "")),
                "delta_log_tail": delta_log[-6:],
                "gap_before_cut_s": (round(time.time() - t0 - last_delta, 1)
                                     if last_delta else None)}

    elapsed = time.time() - t0
    text = "".join(text_parts)
    tool_json = tool_partial.get("_json", "")

    # 从累积的部分 JSON 里提取字段长度（可能是合法 JSON，也可能被截断）
    field_lens = {}
    parsed = None
    if tool_json:
        try:
            parsed = json.loads(tool_json)
            if isinstance(parsed, dict):
                field_lens = {k: len(str(v)) for k, v in parsed.items()}
        except json.JSONDecodeError:
            field_lens = {"_unparsed_len": len(tool_json),
                          "_tail": tool_json[-120:]}
    had_tool = bool(tool_json) or any(t[1] == "tool_use" for t in tool_blocks)

    return {
        "max_tokens": max_tokens, "elapsed_s": round(elapsed, 1),
        "stop_reason": stop, "out_tokens": out_tokens, "in_tokens": in_tokens,
        "first_event_s": round(first_event, 1) if first_event else None,
        "first_delta_s": round(first_delta, 1) if first_delta else None,
        "last_delta_s": round(last_delta, 1) if last_delta else None,
        "n_deltas": n_deltas,
        "text_chars": len(text), "had_tool_use": had_tool,
        "field_lens": field_lens or None,
        "tool_json_complete": parsed is not None,
        "blocks": [list(b) for b in tool_blocks],
        "text_head": text[:400],
        "delta_log_tail": delta_log[-6:],
        "tok_s": round((out_tokens or 0) / elapsed, 1) if elapsed > 0 else None,
    }


def main():
    import pandas as pd
    from anthropic import Anthropic
    from llm_backend import build_user_prompt

    row = int(sys.argv[1]) if len(sys.argv) > 1 else 3579
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    if len(sys.argv) > 3:
        limits = [int(x) for x in sys.argv[3].split(",")]
    else:
        limits = [x for x in (16384, 24576, 32768, 49152, 65536) if x >= cap]

    task_dir = os.path.join(REPO, "rollouts", f"repro_row{row}")
    df = pd.read_parquet(os.path.join(REPO, "dataset/data.parquet"))
    with open(os.path.join(task_dir, "model.py"), "w") as f:
        f.write(df.iloc[row]["code"])
    user = build_user_prompt(task_dir, "", "")

    client = Anthropic(timeout=1800.0)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    print(f"row {row} model={model} prompt={len(user)} chars", flush=True)
    print(f"sweep: {limits}", flush=True)

    results = []
    for mt in limits:
        print(f"\n--- max_tokens={mt} ---", flush=True)
        r = one_call(client, model, user, mt)
        results.append(r)
        if "error" in r:
            print(f"  ERROR after {r['elapsed_s']}s: {r['error']}", flush=True)
            # 传输错误不等于截断，继续试下一级没有意义；直接停
            break
        print(f"  {r['elapsed_s']}s stop={r['stop_reason']} "
              f"out={r['out_tokens']} ({r['tok_s']} tok/s) "
              f"text={r['text_chars']}c tool={r['had_tool_use']}",
              flush=True)
        if r["field_lens"]:
            print(f"  fields={r['field_lens']}", flush=True)
        if r["text_head"]:
            print(f"  text head: {r['text_head'][:200]}", flush=True)
        # 完成 = stop=end_turn 且工具 JSON 解析成功
        if r["stop_reason"] == "end_turn" and r["tool_json_complete"]:
            print(f"  --> {mt} 足够（end_turn + 完整工具调用）", flush=True)
            break
        if r["stop_reason"] == "end_turn" and not r["had_tool_use"]:
            print(f"  --> end_turn 但无工具调用：模型拒绝写文件，不是截断",
                  flush=True)
            break

    out_path = os.path.join(REPO, "rollouts", "stream_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
