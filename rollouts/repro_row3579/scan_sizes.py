#!/usr/bin/env python3
"""A2: find the size where axpby compute beats launch overhead on C500.

The RL reward is speedup vs torch.compile. At 1x128 the kernel is pure launch
overhead and the reward is noise. Sweep n to find the floor where the extension
separates from baseline/compile.
"""
import json
import time

import torch

import cuda_extension


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us


def main():
    torch.manual_seed(0)
    sizes = [128, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18,
             1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24]
    rows = []
    for n in sizes:
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        ref = 2.0 * a + b
        ext = cuda_extension.axpby_forward(a, b, 2.0, 0)

        eager = bench(lambda: torch.add(torch.mul(a, 2.0), b))
        cm = torch.compile(lambda x, y: torch.add(torch.mul(x, 2.0), y))
        _ = cm(a, b)
        compile_us = bench(lambda: cm(a, b))
        ext_us = bench(lambda: cuda_extension.axpby_forward(a, b, 2.0, 0))

        traffic = 3 * 4 * n  # read a, read b, write out
        bw = (traffic / 1e9) / (ext_us / 1e6) if ext_us > 0 else 0
        ok = torch.allclose(ext, ref, atol=1e-4, rtol=1e-4)
        rows.append(dict(
            n=n, eager_us=eager, compile_us=compile_us, ext_us=ext_us,
            speedup_vs_eager=eager / ext_us if ext_us else 0,
            speedup_vs_compile=compile_us / ext_us if ext_us else 0,
            achieved_bw_gbs=bw, correct=ok,
        ))
        print(f"n={n:>9d} eager={eager:8.3f}us compile={compile_us:8.3f}us "
              f"ext={ext_us:8.3f}us  vs_compile={compile_us/ext_us:6.2f}x  "
              f"BW={bw:6.1f}GB/s  ok={ok}")
    with open("/data/cuda-agent-migration/a2_size_scan.json", "w") as f:
        json.dump(rows, f, indent=2)
    # floor: where ext beats compile by a stable margin
    floor = None
    for r in rows:
        if r["speedup_vs_compile"] >= 1.0 and r["n"] >= 1 << 12:
            floor = r["n"]
            break
    print(f"\nsmallest n where ext >= compile: {floor}")


if __name__ == "__main__":
    main()
