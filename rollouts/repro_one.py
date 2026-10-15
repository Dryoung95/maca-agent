#!/usr/bin/env python3
"""复现 collect_rollouts 在 row 3579 上的失败。

批量脚本把任务的 model.py 写进干净的工作区，然后调用 make_attempt_from_llm。
我之前的复现传了一个不存在的路径，所以卡住了。这里按完全相同的方式构建。
"""
import os
import shutil
import sys

REPO = "/data/maca-agent"
sys.path.insert(0, os.path.join(REPO, "agent_workdir"))
os.chdir(REPO)


def build(row_id, code, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    def ignore(dp, names):
        return {n for n in names
                if n in ("build", "__pycache__", "cuda_extension.so")}

    shutil.copytree(os.path.join(REPO, "agent_workdir"), out_dir,
                    ignore=ignore)
    with open(os.path.join(out_dir, "model.py"), "w") as f:
        f.write(code)
    kd = os.path.join(out_dir, "kernels")
    shutil.rmtree(kd, ignore_errors=True)
    os.makedirs(kd)
    p = os.path.join(out_dir, "model_new.py")
    if os.path.exists(p):
        os.remove(p)
    return out_dir


def main():
    import pandas as pd
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 3579
    task_dir = os.path.join(REPO, "rollouts", f"repro_row{row}")

    df = pd.read_parquet(os.path.join(REPO, "dataset/data.parquet"))
    code = df.iloc[row]["code"]
    build(row, code, task_dir)
    print(f"workspace ready: {task_dir}")
    print(f"model.py 头 8 行:")
    for ln in code.splitlines()[:8]:
        print("   ", ln)

    from llm_backend import make_attempt_from_llm, call_model, build_user_prompt, SYSTEM_PROMPT
    user = build_user_prompt(task_dir, "", "")
    print(f"\nprompt 长度: {len(user)} 字符")
    print(f"SKILL.md 大小: {os.path.getsize(os.path.join(task_dir, 'SKILL.md'))} bytes")

    # 单轮，看模型实际返回什么
    import os as _os
    resp = call_model([{"role": "user", "content": user}], SYSTEM_PROMPT,
                      model=_os.environ.get("ANTHROPIC_MODEL"))
    print(f"\nstop_reason: {resp.stop_reason}")
    print(f"usage: {resp.usage.input_tokens} in, {resp.usage.output_tokens} out")
    for b in resp.content:
        t = getattr(b, "type", "?")
        if t == "text":
            print(f"  [text] {b.text[:500]}")
        elif t == "tool_use":
            keys = list(b.input.keys()) if isinstance(b.input, dict) else "?"
            print(f"  [tool_use] name={b.name} keys={keys}")
            if isinstance(b.input, dict):
                for k, v in b.input.items():
                    print(f"      {k}: {len(str(v))} chars")
        elif t == "thinking":
            print(f"  [thinking] {len(b.thinking)} chars")
        else:
            print(f"  [{t}] {str(b)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
