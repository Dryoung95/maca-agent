#!/usr/bin/env python3
"""A1: dataset memory budget grading — CPU only, zero GPU allocation.

Handles module-level constants (batch_size, in_channels, ...) by evaluating
assignments in a restricted namespace, then resolving tensor factory calls
inside get_inputs() through that namespace.
"""
import ast
import json

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

FACTORY = {"randn", "ones", "zeros", "rand", "randint", "full", "arange", "empty"}


def collect_const_ns(tree):
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                v = ast.literal_eval(node.value)
            except Exception:
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    ns[t.id] = v
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                v = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(node.target, ast.Name):
                ns[node.target.id] = v
    return ns


def resolve_dim(a, ns, consts):
    try:
        v = ast.literal_eval(a)
        return int(v)
    except Exception:
        pass
    if isinstance(a, ast.Name) and a.id in ns:
        return int(ns[a.id])
    if isinstance(a, ast.BinOp):
        try:
            l = resolve_dim(a.left, ns, consts)
            r = resolve_dim(a.right, ns, consts)
            if isinstance(a.op, ast.Mult):
                return l * r
            if isinstance(a.op, ast.Add):
                return l + r
        except Exception:
            pass
    if isinstance(a, ast.Attribute):
        try:
            base = resolve_dim(a.value, ns, consts)
            if base is not None:
                return base
        except Exception:
            pass
    return None


def shapes_in(fn_node, ns, consts):
    out = []
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in FACTORY and node.args:
                dims = [resolve_dim(a, ns, consts) for a in node.args]
                if all(d is not None for d in dims):
                    out.append((name, tuple(dims)))
    return out


def dtype_bytes(code, fn_in):
    for node in ast.walk(fn_in):
        if isinstance(node, ast.Call):
            for kw in getattr(node, "keywords", []):
                if kw.arg == "dtype":
                    s = ast.dump(kw.value)
                    for k, b in (("float64", 8), ("double", 8),
                                 ("float32", 4), ("float", 4),
                                 ("float16", 2), ("half", 2),
                                 ("bfloat16", 2), ("int64", 8), ("int32", 4),
                                 ("bool", 1)):
                        if k in s:
                            return b
    if "float64" in code or "double()" in code or "torch.double" in code:
        return 8
    if "float16" in code or "half()" in code or "torch.half" in code:
        return 2
    return 4


def param_shapes(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if name in FACTORY or name == "Parameter":
                dims = []
                ok = True
                for a in n.args:
                    try:
                        dims.append(int(ast.literal_eval(a)))
                    except Exception:
                        ok = False
                if ok and dims:
                    out.append(tuple(dims))
    return out


def main():
    df = pd.read_parquet("/data/cuda-agent-migration/dataset/data.parquet")
    rows = []
    for i in tqdm(df.index, desc="mem budget"):
        code = df.at[i, "code"]
        r = {"row": int(i), "data_source": df.at[i, "data_source"]}
        try:
            tree = ast.parse(code)
            ns = collect_const_ns(tree)
            fn_in = next((n for n in tree.body
                          if isinstance(n, ast.FunctionDef) and n.name == "get_inputs"), None)
            fn_init = next((n for n in tree.body
                            if isinstance(n, ast.FunctionDef)
                            and n.name == "get_init_inputs"), None)
            consts = {}
            if fn_in is None:
                shapes = []
            else:
                shapes = shapes_in(fn_in, ns, consts)
            b = dtype_bytes(code, fn_in if fn_in else tree)
            in_bytes = 0
            for _, dims in shapes:
                e = 1
                for d in dims:
                    e *= d
                in_bytes += e * b
            p = 0
            for dims in param_shapes(tree):
                e = 1
                for d in dims:
                    e *= d
                p += e
            param_bytes = p * b
            # 3x: input + ref output + new output (new may allocate intermediates)
            est_bytes = (in_bytes * 2 + param_bytes) * 3
            if in_bytes == 0 and param_bytes == 0:
                est_bytes = 0
            r.update(parsed=bool(shapes or param_bytes(tree)),
                     in_bytes=in_bytes, param_bytes=param_bytes,
                     est_bytes=est_bytes,
                     in_shapes=json.dumps([s[1] for s in shapes]),
                     b_per_elem=b)
        except Exception as e:
            r.update(parsed=False, in_bytes=0, param_bytes=0, est_bytes=0,
                     in_shapes="", b_per_elem=0, err=type(e).__name__)
        rows.append(r)

    out = pd.DataFrame(rows)
    out["grade"] = out.apply(
        lambda r: "unparsed" if not r.parsed
        else "zero_est" if r.est_bytes == 0
        else "le2GB" if r.est_bytes <= 2e9
        else "le6GB" if r.est_bytes <= 6e9
        else "le12GB" if r.est_bytes <= 12e9
        else "over16GB", axis=1)
    out.to_csv("/data/cuda-agent-migration/dataset/mem_budget.csv", index=False)
    print("\n=== grade counts ===")
    print(out.grade.value_counts())
    print("\n=== est_bytes (nonzero) ===")
    nz = out[out.est_bytes > 0]
    print(nz.est_bytes.describe())
    print("\n=== top-10 largest ===")
    print(out.nlargest(10, "est_bytes")[["row", "in_bytes", "param_bytes", "est_bytes", "grade"]])


if __name__ == "__main__":
    main()
