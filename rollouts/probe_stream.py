#!/usr/bin/env python3
"""流式探针：确定网关 45 分钟切断是否是空闲超时，以及真实吞吐 / 真实 token 需求。

背景：非流式 messages.create 三次在 ~45 分钟收到 HTTP 502（run2 2714s、
run3 首次调用、repro_one run4）。非流式在生成结束前不返回任何字节，网关
看到一条空闲 45 分钟的连接就切断。如果切断的机制是网关→上游读超时
（nginx proxy_read_timeout 一类），那么 SSE 事件持续流动会让计时器一直
重置，流式就能绕过。如果切断是硬性的总时长上限，流式同样会死——这两种
假设只能靠实验区分。

本探针每次收到事件就打点，给出：
  1. 流式是否通过这个网关（第一个事件到达的时间）
  2. 实时吞吐 tok/s
  3. 最终 stop_reason / usage.output_tokens —— 模型到底需要多少 token
     （这是之前始终没能测到的量；8192 被验证会截断，16384 从未完成过）
  4. 数据持续流动时 45 分钟切断是否仍触发
"""
import json
import os
import sys
import time

REPO = "/data/maca-agent"
sys.path.insert(0, os.path.join(REPO, "agent_workdir"))
os.chdir(REPO)


def main():
    import pandas as pd
    from anthropic import Anthropic
    from llm_backend import build_user_prompt, SYSTEM_PROMPT

    row = int(sys.argv[1]) if len(sys.argv) > 1 else 3579
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    task_dir = os.path.join(REPO, "rollouts", f"repro_row{row}")
    if not os.path.isdir(task_dir):
        print(f"工作区不存在: {task_dir}", flush=True)
        return 1

    df = pd.read_parquet(os.path.join(REPO, "dataset/data.parquet"))
    code = df.iloc[row]["code"]
    with open(os.path.join(task_dir, "model.py"), "w") as f:
        f.write(code)
    user = build_user_prompt(task_dir, "", "")

    client = Anthropic(timeout=1800.0)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    print(f"row {row} max_tokens={max_tokens} model={model}", flush=True)
    print(f"prompt {len(user)} chars, SKILL.md "
          f"{os.path.getsize(os.path.join(task_dir, 'SKILL.md'))} bytes",
          flush=True)

    t0 = time.time()
    n_input = n_output = 0
    first_event_at = None
    text_chars = 0
    tool_input = {}
    last_kind = None
    saw_ping = False

    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=[{
                "name": "write_files",
                "description": ("Write the kernel, binding, and model_new.py "
                                "files for this attempt. Call exactly once."),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kernel_cu": {"type": "string"},
                        "binding_cpp": {"type": "string"},
                        "model_new_py": {"type": "string"},
                        "kernel_name": {"type": "string"},
                    },
                    "required": ["kernel_cu", "binding_cpp",
                                 "model_new_py", "kernel_name"],
                },
            }],
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for event in stream:
                now = time.time()
                if first_event_at is None:
                    first_event_at = now - t0
                    print(f"  [stream 首个事件 {first_event_at:.1f}s "
                          f"类型={event.type}]", flush=True)
                kind = event.type
                if kind != last_kind and kind != "content_block_delta":
                    print(f"  [{now-t0:7.1f}s] {kind}", flush=True)
                    last_kind = kind

                if kind == "message_start":
                    n_input = event.message.usage.input_tokens
                elif kind == "content_block_start":
                    b = event.content_block
                    print(f"    block: type={b.type} "
                          f"name={getattr(b, 'name', '-')}", flush=True)
                elif kind == "content_block_delta":
                    d = event.delta
                    dt = getattr(d, "type", "?")
                    if dt == "text_delta":
                        text_chars += len(d.text)
                    elif dt == "input_json_delta":
                        # 部分 JSON，累计后最后再解析
                        tool_input.setdefault("_partial", "")
                        tool_input["_partial"] += d.partial_json
                    n_output += 1  # delta 计数，粗略代理进度
                elif kind == "message_delta":
                    u = event.usage
                    if hasattr(u, "output_tokens"):
                        n_output = u.output_tokens
                    print(f"    message_delta: usage={u} "
                          f"stop={event.delta.stop_reason}", flush=True)

                # 每 120 秒打一次进度
                if int(now - t0) % 120 == 0 and int(now - t0) > 0:
                    pass  # 由下面 elapsed 日志覆盖

            final = stream.get_final_message()
            elapsed = time.time() - t0
            print(f"\n=== 完成 {elapsed:.1f}s ({elapsed/60:.1f} min) ===",
                  flush=True)
            print(f"stop_reason: {final.stop_reason}", flush=True)
            print(f"input_tokens: {final.usage.input_tokens}", flush=True)
            print(f"output_tokens: {final.usage.output_tokens}", flush=True)
            if elapsed > 60 and final.usage.output_tokens > 0:
                print(f"平均吞吐: "
                      f"{final.usage.output_tokens/elapsed:.1f} tok/s",
                      flush=True)
            kinds = [getattr(b, "type", "?") for b in final.content]
            print(f"content blocks: {kinds}", flush=True)
            tu = [b for b in final.content
                  if getattr(b, "type", None) == "tool_use"]
            if tu:
                inp = tu[0].input
                print(f"write_files: YES name={tu[0].name}", flush=True)
                for k, v in inp.items():
                    print(f"    {k}: {len(str(v))} chars", flush=True)
            else:
                print("write_files: NO —— 模型没有产出工具调用", flush=True)
                for b in final.content:
                    if getattr(b, "type", None) == "text":
                        print(f"  [text {len(b.text)} chars] "
                              f"{b.text[:600]}", flush=True)
            print(f"text_chars={text_chars} first_event={first_event_at:.1f}s",
                  flush=True)

            # 保存完整响应用于诊断
            out = {
                "row": row, "max_tokens": max_tokens,
                "elapsed_s": elapsed, "stop_reason": final.stop_reason,
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
                "blocks": kinds, "has_write_files": bool(tu),
                "first_event_s": first_event_at,
                "text_chars": text_chars,
                "tool_field_lens": ({k: len(str(v)) for k, v in tu[0].input.items()}
                                    if tu else None),
                "text": "".join(b.text for b in final.content
                                if getattr(b, "type", None) == "text")[:4000],
            }
            with open(os.path.join(
                    REPO, "rollouts", "stream_probe_result.json"), "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print("-> rollouts/stream_probe_result.json", flush=True)
            return 0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n=== 异常 {elapsed:.1f}s ({elapsed/60:.1f} min) ===",
              flush=True)
        print(f"{type(e).__name__}: {str(e)[:500]}", flush=True)
        print(f"first_event_at={first_event_at} text_chars={text_chars} "
              f"deltas~{n_output}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
