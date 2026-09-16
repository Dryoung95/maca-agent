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
CFG_BY_KERNEL = {"axpby_forward": (0, 1, 2), "axpby_v2_forward": (0, 1, 2),
                 "axpby_v3_forward": (0, 1, 2, 3)}
rows = []
for N in [1 << 16, 1 << 18, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24]:
    a = torch.randn(N, device="cuda"); b = torch.randn(N, device="cuda")
    sink = torch.empty(N, device="cuda")
    eager = bench(lambda: torch.add(torch.mul(a, 2.0), b, out=sink))
    cm = torch.compile(lambda x, y: torch.add(torch.mul(x, 2.0), y))
    _ = cm(a, b)
    compile_us = bench(lambda: cm(a, b))
    r = {"n": N, "eager": eager, "compile": compile_us}
    for k, cfgs in CFG_BY_KERNEL.items():
        fn = getattr(cuda_extension, k)
        best, bc = None, 0
        for cfg in cfgs:
            t = bench(lambda: fn(a, b, 2.0, cfg))
            if best is None or t < best:
                best, bc = t, cfg
        r[k] = best
        r[k + "_cfg"] = bc
    r["v3_vs_compile"] = compile_us / r["axpby_v3_forward"]
    r["v3_vs_v2"] = r["axpby_v2_forward"] / r["axpby_v3_forward"]
    rows.append(r)
    print(f"n={N:>9d} compile={compile_us:7.1f} v1={r['axpby_forward']:7.1f} "
          f"v2={r['axpby_v2_forward']:7.1f} v3={r['axpby_v3_forward']:7.1f} "
          f"| v3/compile={r['v3_vs_compile']:5.2f}x v3/v2={r['v3_vs_v2']:5.2f}x")
json.dump(rows, open("/data/cuda-agent-migration/b2_kernel_cmp.json", "w"), indent=2)
