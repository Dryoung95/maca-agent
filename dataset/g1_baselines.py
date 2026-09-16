"""G1: 5 个纯 elementwise 融合任务的 baseline 探测。

对每个任务：构造 Model、生成输入、跑 eager / torch.compile，记录显存与时间。
随后由 g1_run.py 走 KernelBench 官方门槛。
"""
import json
import time

import pandas as pd
import torch

TARGETS = [2885, 1573, 480, 494, 311]


def probe(row_id):
    df = pd.read_parquet("dataset/data.parquet")
    code = df.iloc[row_id]["code"]
    ns = {}
    exec(compile(code, f"<row{row_id}>", "exec"), ns)
    Model = ns["Model"]
    get_inputs = ns["get_inputs"]
    get_init_inputs = ns["get_init_inputs"]

    model = Model(*get_init_inputs()).cuda()
    torch.cuda.reset_peak_memory_stats()
    inputs = [x.cuda() for x in get_inputs()]

    with torch.no_grad():
        for _ in range(10):
            ref = model(*inputs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            ref = model(*inputs)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / 50 * 1e6

    peak = torch.cuda.max_memory_allocated() / 1e9

    compile_us = None
    try:
        cmodel = torch.compile(model)
        with torch.no_grad():
            for _ in range(10):
                cmodel(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                cmodel(*inputs)
        torch.cuda.synchronize()
        compile_us = (time.perf_counter() - t0) / 50 * 1e6
    except Exception as e:
        print(f"  compile failed: {type(e).__name__}: {e}")

    print(f"row {row_id}: out shape {tuple(ref.shape)} dtype {ref.dtype} "
          f"nelem {ref.numel()/1e6:.1f}M")
    print(f"  eager {eager_us:.1f}us | compile {compile_us}us | peak {peak:.2f}GB")
    return {
        "row": row_id,
        "ops": str(df.iloc[row_id]["ops"]),
        "out_shape": list(ref.shape),
        "out_dtype": str(ref.dtype),
        "nelem": ref.numel(),
        "eager_us": eager_us,
        "compile_us": compile_us,
        "peak_gb": peak,
    }


def main():
    results = []
    for rid in TARGETS:
        print(f"=== row {rid} ===")
        try:
            results.append(probe(rid))
        except Exception as e:
            print(f"row {rid} FAILED: {type(e).__name__}: {e}")
    with open("g1_baselines.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
