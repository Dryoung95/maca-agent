"""G4b: 压力测试「静态估算 > 16.3GB 硬上限」的 downsize 任务。

G4 回归发现 row 3228（est 38.65GB，batch 512→132）降档后仍 OOM。这类任务
在 actionable_tasks.csv 里有 125 条（占 downsize 档 52%）。

E3 已证明静态估算系统性偏高 2–6×，所以这些任务降档后**可能**能跑——但既然
估算不可信，「3396 条可用」这个池子规模就没有依据。本脚本抽样实测，用
max_memory_allocated() 而非静态估算，给出降档后的真实可用率。
"""
import gc
import json
import random
import re
import sys
import time

import pandas as pd
import torch

SAMPLE = 12
SEED = 7
MIN_FREE_GB = 4.0


def rewrite_batch(code, name, new_val):
    pat = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)(\d+)", re.M)
    new, n = pat.subn(rf"\g<1>{new_val}", code)
    return new, n


def free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def probe(df, row_id, batch_to):
    code = df.iloc[row_id]["code"]
    # 批量常量名按出现顺序取第一个（与 downsize_analysis 一致）
    names = re.findall(r"^(\w+)\s*=\s*\d+\s*(#.*)?$", code, re.M)
    # 优先尝试常见的 batch 名
    for cand in ("batch_size", "batch", "B", "bs"):
        new_code, n = rewrite_batch(code, cand, batch_to)
        if n >= 1:
            break
    else:
        # 退而用第一个模块级 int 常量
        m = re.search(r"^(\w+)\s*=\s*(\d+)\s*$", code, re.M)
        if not m:
            return {"ok": False, "reason": "no batch constant found"}
        new_code, n = rewrite_batch(code, m.group(1), batch_to)
        if n < 1:
            return {"ok": False, "reason": "rewrite failed"}

    ns = {}
    exec(compile(new_code, f"<row{row_id}>", "exec"), ns)
    model = ns["Model"](*ns["get_init_inputs"]()).eval().cuda()
    torch.cuda.reset_peak_memory_stats()
    inputs = [t.cuda() for t in ns["get_inputs"]()]
    with torch.no_grad():
        out = model(*inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, inputs, out, ns
    gc.collect()
    torch.cuda.empty_cache()
    return {"ok": True, "peak_gb": round(peak, 3),
            "batch_to": batch_to}


def main():
    actionable = pd.read_csv("dataset/actionable_tasks.csv")
    df = pd.read_parquet("dataset/data.parquet")

    over = actionable[(actionable["strategy"] == "downsize")
                      & (actionable["est_gb"] > 16.3)]
    print(f"静态估算 > 16.3GB 的 downsize 任务: {len(over)} 条")
    sample = over.sample(min(SAMPLE, len(over)), random_state=SEED)

    results = []
    for _, a in sample.iterrows():
        row_id = int(a["row"])
        if free_gb() < MIN_FREE_GB:
            print(f"[ABORT] free={free_gb():.2f}GB — 让出 GPU")
            break
        t0 = time.time()
        rec = {"row": row_id, "est_gb": a["est_gb"],
               "batch": a["batch"], "batch_to": a["batch_to"]}
        try:
            rec.update(probe(df, row_id, int(a["batch_to"])))
        except torch.cuda.OutOfMemoryError as e:
            rec.update({"ok": False, "reason": f"OOM: {str(e)[:80]}"})
        except Exception as e:
            rec.update({"ok": False,
                        "reason": f"{type(e).__name__}: {str(e)[:80]}"})
        rec["wall_s"] = round(time.time() - t0, 1)
        results.append(rec)
        status = (f"peak {rec['peak_gb']:.2f}GB" if rec.get("ok")
                  else rec.get("reason", "?"))
        print(f"row {row_id:>5}: est {rec['est_gb']:6.2f}GB "
              f"batch {rec['batch']:>5.0f}->{rec['batch_to']:>5.0f} "
              f"| {status} ({rec['wall_s']}s)")
        time.sleep(1)

    ok = [r for r in results if r.get("ok")]
    print(f"\n=== G4b: {len(ok)}/{len(results)} 降档后可执行 ===")
    if ok:
        print(f"峰值显存: min {min(r['peak_gb'] for r in ok):.2f} / "
              f"max {max(r['peak_gb'] for r in ok):.2f} GB")
    with open("g4_downsize_stress.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
