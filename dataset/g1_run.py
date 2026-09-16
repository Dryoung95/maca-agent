"""G1: KernelBench 官方门槛判定（eval_kernel_against_ref）对 5 个 G1 任务。

关键：eval 用 load_custom_model_with_tempfile 在隔离上下文加载 model_new.py，
工作目录不是 task dir，必须把 task 目录放进 PYTHONPATH，否则
`ModuleNotFoundError: cuda_extension` -> 假阴性 compiled=False（本平台已知衔接点）。
"""
import json
import os
import sys

from kernelbench.eval import eval_kernel_against_ref

TARGETS = [2885, 1573, 480, 494, 311]
BASE = "/data/cuda-agent-migration/tasks"


def main():
    results = []
    for rid in TARGETS:
        task_dir = os.path.join(BASE, f"row{rid}")
        original = open(os.path.join(task_dir, "model.py")).read()
        custom = open(os.path.join(task_dir, "model_new.py")).read()
        # 每个任务有自己的 cuda_extension.so；必须丢弃上一轮缓存的模块，
        # 否则 import 拿到的是另一个 task 的 .so（AttributeError 假阴性）
        sys.modules.pop("cuda_extension", None)
        sys.path.insert(0, task_dir)
        print(f"=== row {rid} ===", flush=True)
        try:
            res = eval_kernel_against_ref(
                original_model_src=original,
                custom_model_src=custom,
                measure_performance=True,
                timing_method="cuda_event",
                verbose=True,
            )
            d = res.to_dict() if hasattr(res, "to_dict") else vars(res)
            d = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                     else str(v)) for k, v in d.items()}
            d["row"] = rid
            results.append(d)
            print(f"row {rid}: {d}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"row": rid, "error": f"{type(e).__name__}: {e}"})
        finally:
            try:
                sys.path.remove(task_dir)
            except ValueError:
                pass

    with open("g1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
