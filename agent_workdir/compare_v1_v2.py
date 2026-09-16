import json, time
import torch
import cuda_extension

def bench(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (t0 - time.perf_counter()) / -n * 1e6

torch.manual_seed(0)
rows = []
for N in [1 << 16, 1 << 18, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24]:
    a = torch.randn(N, device="cuda"); b = torch.randn(N, device="cuda")
    sink = torch.empty(N, device="cuda")
    eager = bench(lambda: torch.add(torch.mul(a, 2.0), b, out=sink))
    cm = torch.compile(lambda x, y: torch.add(torch.mul(x, 2.0), y))
    _ = cm(a, b)
    compile_us = bench(lambda: cm(a, b))
    v1 = bench(lambda: cuda_extension.axpby_forward(a, b, 2.0, 0))
    best_v2, best_cfg = None, 0
    for cfg in (0, 1, 2):
        t = bench(lambda: cuda_extension.axpby_v2_forward(a, b, 2.0, cfg))
        if best_v2 is None or t < best_v2:
            best_v2, best_cfg = t, cfg
    r = dict(n=N, eager=eager, compile=compile_us, v1=v1, v2=best_v2, cfg=best_cfg,
             v2_vs_compile=compile_us / best_v2, v2_vs_v1=v1 / best_v2)
    rows.append(r)
    print(f"n={N:>9d} eager={eager:7.1f} compile={compile_us:7.1f} v1={v1:7.1f} "
          f"v2={best_v2:7.1f} (cfg{best_cfg})  v2/compile={compile_us/best_v2:5.2f}x  v2/v1={v1/best_v2:5.2f}x")
json.dump(rows, open("/data/cuda-agent-migration/b1_kernel_cmp.json", "w"), indent=2)
