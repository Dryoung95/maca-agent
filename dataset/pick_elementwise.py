"""Pick elementwise-fusion candidates from actionable_tasks.csv.

甜点任务分型（G2）：ops 全部是纯一元 elementwise（无 reduce / matmul / conv /
布局变化），输出 float。F2a/E4 的成功模式。用精确白名单匹配，避免子串误伤。
"""
import json

import pandas as pd

# 一元 elementwise（输出形状 = 输入形状，输出 float）
UNARY = {
    "torch.abs", "torch.acos", "torch.asin", "torch.atan", "torch.ceil",
    "torch.cos", "torch.cosh", "torch.exp", "torch.floor", "torch.frac",
    "torch.neg", "torch.reciprocal", "torch.round", "torch.rsqrt",
    "torch.sigmoid", "torch.sign", "torch.sin", "torch.sinh", "torch.sqrt",
    "torch.tan", "torch.tanh", "torch.trunc", "torch.log", "torch.log2",
    "torch.log10", "torch.log1p", "torch.erf", "torch.erfc", "torch.expm1",
    "torch.relu", "torch.silu", "torch.gelu", "torch.tanh",
    "torch.leaky_relu", "torch.elu", "torch.softplus", "torch.mish",
    "torch.hardsigmoid", "torch.hardswish", "torch.hardtanh",
    "torch.logsigmoid", "torch.celu", "torch.selu", "torch.rrelu",
}
# 二元 pointwise（形状相同或广播）—— 允许，但需要至少一个一元 op
BINARY = {
    "torch.add", "torch.sub", "torch.mul", "torch.div", "torch.pow",
    "torch.fmod", "torch.remainder", "torch.maximum", "torch.minimum",
    "torch.atan2", "torch.copysign", "torch.fmin", "torch.fmax", "torch.lerp",
}
# nn.* 激活（无参数或参数不改变形状）—— 允许
NN_ACTIVATION = {
    "nn.ReLU", "nn.Sigmoid", "nn.Tanh", "nn.GELU", "nn.SiLU", "nn.ELU",
    "nn.LeakyReLU", "nn.Softplus", "nn.Mish", "nn.Hardsigmoid",
    "nn.Hardswish", "nn.Hardtanh", "nn.LogSigmoid", "nn.CELU", "nn.SELU",
    "nn.RReLU", "nn.PReLU", "nn.Softsign", "nn.Softshrink",
    "nn.Tanhshrink", "nn.Threshold", "nn.Softmin",
}
ALL_ALLOWED = UNARY | BINARY | NN_ACTIVATION


def parse_ops(ops_field):
    if isinstance(ops_field, str):
        try:
            return json.loads(ops_field)
        except json.JSONDecodeError:
            return [ops_field]
    if isinstance(ops_field, (list, tuple)):
        return list(ops_field)
    return []


def main():
    actionable = pd.read_csv("dataset/actionable_tasks.csv")
    df = pd.read_parquet("dataset/data.parquet")
    actionable = actionable[actionable["strategy"] == "direct"].copy()

    rows = []
    for _, a in actionable.iterrows():
        row_id = int(a["row"])
        ops = parse_ops(df.iloc[row_id]["ops"])
        if not ops:
            continue
        if not all(o in ALL_ALLOWED for o in ops):
            continue
        unary = [o for o in ops if o in UNARY or o in NN_ACTIVATION]
        if not unary:
            continue
        rows.append({
            "row": row_id,
            "ops": "|".join(ops),
            "n_ops": len(ops),
            "n_unary": len(unary),
            "unary": "|".join(unary),
            "est_gb": a["est_gb"],
            "grade": a["grade"],
        })

    out = pd.DataFrame(rows).sort_values(
        ["n_unary", "est_gb"], ascending=False)
    print(f"pure-elementwise direct candidates: {len(out)}")
    if len(out):
        print(out.head(30).to_string(index=False))
        out.to_csv("dataset/g1_candidates.csv", index=False)


if __name__ == "__main__":
    main()
