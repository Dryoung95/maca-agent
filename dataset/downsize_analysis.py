#!/usr/bin/env python3
"""E2: can the over-16GB tasks be downsized to fit the C500's 16.3 GB?

For each task whose static estimate exceeds 16 GB, find the batch_size
constant in the code and compute the largest batch whose estimate still fits
under a safety budget. Pure AST work — no GPU allocation.
"""
import ast
import json

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

BUDGET_GB = 10.0   # leave headroom: 16.3 total, other agents may use the card


def find_batch_assigns(tree):
    """Return {name: value} for module-level integer assignments whose name
    looks like a batch dimension."""
    out = {}
    for node in tree.body:
        val = None
        if isinstance(node, ast.Assign):
            try:
                val = int(ast.literal_eval(node.value))
            except Exception:
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = val
    return out


def main():
    bud = pd.read_csv(os.path.join(paths.DATASET_DIR, "mem_budget.csv"))
    over = bud[(bud.grade == "over16GB") & (bud.est_bytes > 0)]
    df = paths.load_dataframe()

    rows = []
    for _, r in over.iterrows():
        row = int(r["row"])
        code = df.at[row, "code"]
        est = r["est_bytes"]
        rec = {"row": row, "est_gb": round(est / 1e9, 2),
               "b_per_elem": r["b_per_elem"]}
        try:
            tree = ast.parse(code)
            consts = find_batch_assigns(tree)
            # which const drives the batch? Look for the first one referenced
            # in get_inputs' first tensor shape
            fn_in = next((n for n in tree.body
                          if isinstance(n, ast.FunctionDef) and n.name == "get_inputs"), None)
            batch_name, batch_val = None, None
            if fn_in is not None:
                for node in ast.walk(fn_in):
                    if isinstance(node, ast.Call):
                        for a in node.args:
                            if isinstance(a, ast.Name) and a.id in consts:
                                batch_name, batch_val = a.id, consts[a.id]
                                break
                        if batch_name:
                            break
            rec.update(batch_name=batch_name, batch_val=batch_val)
            # how much of est is proportional to batch? We approximate:
            # est scales linearly with batch (input + intermediates).
            if batch_val and batch_val > 0:
                # est = batch * per_batch + fixed (params)
                param_b = r["param_bytes"]
                per_batch = (est - param_b * 3) / batch_val
                # solve batch_max: per_batch*b + param*3 <= budget
                budget = BUDGET_GB * 1e9
                bmax = int((budget - param_b * 3) / per_batch) if per_batch > 0 else 0
                rec["per_batch_bytes"] = round(per_batch)
                rec["max_batch_for_budget"] = max(0, bmax)
                rec["fits"] = bmax >= 1
        except Exception as e:
            rec["err"] = type(e).__name__
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths.DATASET_DIR, "downsize_analysis.csv"),
               index=False)
    print("over-16GB tasks analyzed:", len(out))
    print("with a batch constant:", out.batch_val.notna().sum())
    print("can be downsized to fit:", (out.fits == True).sum() if "fits" in out else 0)
    if "max_batch_for_budget" in out:
        print("\nmax_batch distribution (for those that fit):")
        f = out[out.fits == True]
        if len(f):
            print(f.max_batch_for_budget.describe())
    print("\nsample:")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
