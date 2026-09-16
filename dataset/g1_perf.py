"""G1 perf: 直接计时各任务的 fused kernel（v1 标量），对照 eager 与带宽上限。

不依赖 KernelBench 的隔离加载，避免 cuda_extension 模块缓存问题；
每个 task 的 .so 用唯一模块名按路径加载。
"""
import importlib.util
import json
import os
import sys
import time

import pandas as pd
import torch

TARGETS = {
    2885: "fused_selu_tanh_forward",
    1573: "fused_log_cos_forward",
    480: "fused_relu_softplus_forward",
    494: "fused_softplus_leaky_relu_forward",
    311: "fused_leaky_relu_leaky_relu_forward",
}
BASE = "/data/cuda-agent-migration/tasks"


def load_ext(rid):
    """加载 task 已编译好的 cuda_extension.so。.so 的导出符号固定为
    PyInit_cuda_extension，所以模块名必须是 'cuda_extension'；每次先清掉
    sys.modules 里的旧条目，实现同进程切换多个 task 的 .so。"""
    so = os.path.join(BASE, f"row{rid}", "cuda_extension.so")
    sys.modules.pop("cuda_extension", None)
    spec = importlib.util.spec_from_file_location("cuda_extension", so)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cuda_extension"] = mod
    spec.loader.exec_module(mod)
    return mod


def time_fn(fn, args, iters=50):
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn(*args)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters * 1000  # us


def main():
    df = pd.read_parquet("dataset/data.parquet")

    # 带宽上限探针：纯 copy（读+写）
    probe = torch.randn(200_000_000, device="cuda")
    dst = torch.empty_like(probe)
    copy_us = time_fn(lambda a, b: b.copy_(a), (probe, dst))
    bw = 2 * probe.numel() * 4 / copy_us / 1e3  # GB/s
    print(f"[probe] copy {copy_us:.0f}us -> {bw:.0f} GB/s (read+write)")
    del probe, dst
    torch.cuda.empty_cache()

    results = []
    for rid, fn_name in TARGETS.items():
        code = df.iloc[rid]["code"]
        ns = {}
        exec(compile(code, f"<row{rid}>", "exec"), ns)
        Model = ns["Model"]
        model = Model(*ns["get_init_inputs"]()).cuda()
        inputs = [x.cuda() for x in ns["get_inputs"]()]

        with torch.no_grad():
            eager_us = time_fn(model, inputs)
            ext = load_ext(rid)
            with torch.no_grad():
                ext_us = time_fn(getattr(ext, fn_name), inputs)
        n = inputs[0].numel()
        traffic_gb = 2 * n * 4 / 1e9
        print(f"row {rid}: n={n/1e6:.0f}M eager {eager_us:.0f}us | ext {ext_us:.0f}us "
              f"| {eager_us/ext_us:.2f}x eager | achieved {traffic_gb/(ext_us/1e6):.0f} GB/s")
        results.append({
            "row": rid,
            "n": n,
            "eager_us": eager_us,
            "ext_v1_us": ext_us,
            "speedup_vs_eager": eager_us / ext_us,
            "achieved_gbs": traffic_gb / (ext_us / 1e6),
            "peak_gbs": bw,
        })
        del inputs, model
        torch.cuda.empty_cache()

    with open("g1_perf.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
