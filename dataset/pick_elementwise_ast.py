"""AST-based sweet-spot task picker.

dataset 的 `ops` 字段只记录 torch.* 函数调用，**漏掉 forward 里构造的 nn 模块**
（nn.Linear / nn.Conv2d 等）。实测：ops=["torch.log","torch.sigmoid"] 的任务
forward 里可以含一个 nn.Linear。因此必须解析源码本身。

判定标准（严格白名单）：forward 里出现的每个 torch.* 调用、以及 __init__ 里构造的
每个 nn 模块，都必须是纯 elementwise 算子；输出 dtype 为 float。
这正是 E4 / F2a 通过门槛的任务分型（G2）。
"""
import ast
import json
from collections import Counter

import pandas as pd

TORCH_ELEMENTWISE = {
    "abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "exp", "floor",
    "frac", "neg", "reciprocal", "round", "rsqrt", "sigmoid", "sign",
    "sin", "sinh", "sqrt", "tan", "tanh", "trunc", "log", "log2", "log10",
    "log1p", "erf", "erfc", "expm1", "relu", "silu", "gelu", "leaky_relu",
    "elu", "softplus", "mish", "hardsigmoid", "hardswish", "hardtanh",
    "logsigmoid", "celu", "selu", "rrelu", "softsign", "softshrink",
    "tanhshrink", "threshold", "deg2rad", "rad2deg", "nan_to_num",
    "positive",
}
TORCH_BINARY = {
    "add", "sub", "mul", "div", "pow", "fmod", "remainder", "maximum",
    "minimum", "atan2", "copysign", "fmin", "fmax", "lerp",
}
NN_ELEMENTWISE = {
    "ReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "ELU", "LeakyReLU",
    "Softplus", "Mish", "Hardsigmoid", "Hardswish", "Hardtanh",
    "LogSigmoid", "CELU", "SELU", "RReLU", "PReLU", "Softsign",
    "Softshrink", "Tanhshrink", "Threshold", "Softmin", "Identity",
}
TORCH_PASSTHROUGH = {"stack", "cat"}  # F2a 证实的可融合特例


class Collector(ast.NodeVisitor):
    """收集 forward 体内出现的 self.<attr>() 调用与 torch/F.<name>() 调用。"""

    def __init__(self):
        self.module_attrs = set()
        self.torch_calls = set()

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id in ("torch", "F"):
                self.torch_calls.add(f.attr)
            elif f.value.id == "self":
                self.module_attrs.add(f.attr)
        elif isinstance(f, ast.Name):
            self.torch_calls.add(f.id)
        self.generic_visit(node)


def _extract_constructor(value):
    """从 __init__ 赋值右端提取被构造的类名（覆盖容器/推导式形态）。"""
    if isinstance(value, ast.Call):
        if isinstance(value.func, ast.Attribute):
            return value.func.attr
        if isinstance(value.func, ast.Name):
            return value.func.id
    if isinstance(value, ast.Dict) and value.values:
        return _extract_constructor(value.values[0])
    if isinstance(value, (ast.List, ast.Tuple)) and value.elts:
        return _extract_constructor(value.elts[0])
    if isinstance(value, ast.ListComp):
        return _extract_constructor(value.elt) or "Unknown"
    return "Unknown"


def analyze(code):
    tree = ast.parse(code)

    forward = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            forward = node
            break
    if forward is None:
        return ["no forward"], False

    coll = Collector()
    for stmt in forward.body:
        coll.visit(stmt)

    # __init__ 里 self.<attr> = <constructor> 的映射
    constructed = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if not (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    continue
                constructed[t.attr] = _extract_constructor(node.value)

    bad = []
    for attr in coll.module_attrs:
        cls = constructed.get(attr, "Unknown")
        if cls not in NN_ELEMENTWISE:
            bad.append(f"self.{attr}=nn.{cls}")
    # __init__ 里构造的所有模块都必须是 elementwise —— 排除 nn.Sequential /
    # ModuleList / ModuleDict 等容器（展开后实际算子类型不可知）
    for attr, cls in constructed.items():
        if cls not in NN_ELEMENTWISE:
            bad.append(f"init self.{attr}=nn.{cls}")
    for name in coll.torch_calls:
        if name in TORCH_ELEMENTWISE or name in TORCH_BINARY:
            continue
        if name in TORCH_PASSTHROUGH:
            continue
        bad.append(f"torch.{name}")

    outputs_bool = any(k in code for k in ("torch.randint", "bool"))
    return bad, outputs_bool


def parse_ops(ops_field):
    if isinstance(ops_field, (list, tuple)):
        return [str(x) for x in ops_field]
    if not isinstance(ops_field, str):
        return []
    s = ops_field.strip()
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except json.JSONDecodeError:
        pass
    s = s.strip("[]")
    return [p.strip().strip("'\"") for p in s.split(",") if p.strip()]


def main():
    actionable = pd.read_csv("dataset/actionable_tasks.csv")
    df = pd.read_parquet("dataset/data.parquet")
    actionable = actionable[actionable["strategy"] == "direct"].copy()

    rows, rejected = [], []
    for _, a in actionable.iterrows():
        row_id = int(a["row"])
        code = df.iloc[row_id]["code"]
        try:
            bad, bool_out = analyze(code)
        except SyntaxError:
            rejected.append((row_id, "syntax"))
            continue
        if bad or bool_out:
            rejected.append((row_id, ";".join(bad) or "bool-out"))
            continue
        ops = parse_ops(df.iloc[row_id]["ops"])
        rows.append({
            "row": row_id,
            "ops": "|".join(ops),
            "n_ops": len(ops),
            "est_gb": a["est_gb"],
            "grade": a["grade"],
        })

    out = pd.DataFrame(rows).sort_values("est_gb")
    print(f"AST-verified pure-elementwise direct tasks: {len(out)}")
    print(out.to_string(index=False))
    out.to_csv("dataset/g1_candidates.csv", index=False)

    c = Counter(r[1].split(";")[0] for r in rejected)
    print(f"\nrejected: {len(rejected)}; top reasons:")
    for k, v in c.most_common(15):
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
