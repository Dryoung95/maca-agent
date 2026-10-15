#!/usr/bin/env python3
"""确认 max_tokens=8192 是 row 3579 失败的根因，并找到足够大的上限。

复现结果：stop_reason=max_tokens，usage=8192 out。
模型还没写完 write_files 就被截断，所以 extract_attempt 连续 3 轮找不到
完整工具调用。axpby 示例能一次过是因为 kernel 短；真实任务带 nn.Softmax
这类参数化 op，输出更长。

这里对同一个任务试几个上限，记录 stop_reason / 输出 token / 是否完整产出
write_files。用 print(..., flush=True) 立即落盘，避免上一次日志全空的情况。
"""
import json
import os
import sys

REPO = "/data/maca-agent"
sys.path.insert(0, os.path.join(REPO, "agent_workdir"))
os.chdir(REPO)


def main():
    import pandas as pd
    from llm_backend import call_model, build_user_prompt, SYSTEM_PROMPT

    row = int(sys.argv[1]) if len(sys.argv) > 1 else 3579
    model = os.environ.get("ANTHROPIC_MODEL")
    df = pd.read_parquet(os.path.join(REPO, "dataset/data.parquet"))
    task_dir = os.path.join(REPO, "rollouts", f"repro_row{row}")
    user = build_user_prompt(task_dir, "", "")

    out_rows = []
    for mt in (8192, 12288, 16384):
        print(f"--- max_tokens={mt} ---", flush=True)
        try:
            r = call_model([{"role": "user", "content": user}], SYSTEM_PROMPT,
                           max_tokens=mt, model=model)
            kinds = [getattr(b, "type", "?") for b in r.content]
            tu = [b for b in r.content
                  if getattr(b, "type", None) == "tool_use"]
            ok = bool(tu)
            keys = list(tu[0].input.keys()) if ok else []
            # 每个字段的长度，看输出是否仍然逼近上限
            lens = {k: len(str(v)) for k, v in
                    (tu[0].input.items() if ok else [])}
            print(f"stop={r.stop_reason} out_tokens={r.usage.output_tokens} "
                  f"blocks={kinds} write_files={'YES' if ok else 'NO'}",
                  flush=True)
            if ok:
                print(f"keys={keys} lens={lens}", flush=True)
            out_rows.append({"max_tokens": mt, "stop": r.stop_reason,
                             "out_tokens": r.usage.output_tokens,
                             "write_files": ok, "field_lens": lens})
            if ok:
                print(f"--> {mt} 已足够，停下", flush=True)
                break
        except Exception as e:
            print(f"FAILED: {type(e).__name__} {str(e)[:200]}", flush=True)
            out_rows.append({"max_tokens": mt, "error": str(e)[:200]})

    with open("/data/maca-agent/rollouts/maxtokens_probe.json", "w") as f:
        json.dump(out_rows, f, indent=2)
    print("-> rollouts/maxtokens_probe.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
