"""G4: 抽样正确性/可执行性回归。

profiling.py 的 DCE 修复和 SKILL.md 分型表更新都改动了环境层。本脚本从
actionable_tasks.csv 抽样 20 条（分层：甜点 / 非甜点 / 降档），确认
eager + torch.compile 双模仍可执行、显存仍在预算内。

注意：actionable_tasks.csv 的 `row` 是 dataset/data.parquet 的行号。
"""
import json
import random
import sys
import time

import pandas as pd
import torch

SAMPLE = 20
SEED = 42


def probe_row(df, row_id):
    code = df.iloc[row_id]["code"]
    ns = {}
    exec(compile(code, f"<row{row_id}>", "exec"), ns)
    Model = ns["Model"]
    get_inputs = ns["get_inputs"]
    get_init_inputs = ns["get_init_inputs"]

    model = Model(*get_init_inputs()).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    inputs = [x.cuda() for x in get_inputs()]

    with torch.no_grad():
        out = model(*inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {"out_shape": list(out.shape) if hasattr(out, "shape") else str(out),
            "peak_gb": peak}


def main():
    actionable = pd.read_csv("dataset/actionable_tasks.csv")
    df = pd.read_parquet("dataset/data.parquet")

    rng = random.Random(SEED)
    direct = actionable[actionable["strategy"] == "direct"]
    downsize = actionable[actionable["strategy"] == "downsize"]

    # 分层抽样：12 direct + 8 downsize，覆盖不同显存档位
    sample = (direct.sample(12, random_state=SEED).to_dict("records")
              + downsize.sample(8, random_state=SEED).to_dict("records"))
    rng.shuffle(sample)

    results, failures = [], []
    for a in sample:
        row_id = int(a["row"])
        t0 = time.time()
        try:
            info = probe_row(df, row_id)
            rec = {"row": row_id, "strategy": a["strategy"],
                   "est_gb": a["est_gb"], "grade": a["grade"],
                   "status": "ok", "wall_s": round(time.time() - t0, 1),
                   **info}
            results.append(rec)
            print(f"row {row_id:>5} [{a['strategy']:>8}] "
                  f"est {a['est_gb']:.2f}GB -> peak {info['peak_gb']:.2f}GB "
                  f"{info['out_shape']} ({rec['wall_s']}s)")
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:120]}"
            failures.append({"row": row_id, "strategy": a["strategy"],
                             "error": msg})
            print(f"row {row_id:>5} [{a['strategy']:>8}] FAILED: {msg}")

    ok = len(results)
    print(f"\n=== G4 回归结果: {ok}/{len(sample)} 可执行, "
          f"{len(failures)} 失败 ===")
    if results:
        peaks = [r["peak_gb"] for r in results]
        print(f"峰值显存: min {min(peaks):.2f} / median "
              f"{sorted(peaks)[len(peaks)//2]:.2f} / max {max(peaks):.2f} GB")
    for f in failures:
        print(f"  FAIL row {f['row']}: {f['error']}")

    with open("g4_regression.json", "w") as f:
        json.dump({"results": results, "failures": failures}, f, indent=2)


if __name__ == "__main__":
    main()
