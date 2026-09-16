# CUDA-Agent → MetaX C500 (MACA) 适配 — 第一阶段报告

**日期**：2026-09-15
**范围**：论文资料下载 + `agent_workdir` 构建闭环 + 正确性闭环 + 与已迁移 KernelBench 的衔接验证。
**不含**：性能调优、RL 训练管线、大规模数据集跑批。

---

## 1. 下载清单

| 资源 | 位置 | 状态 |
|---|---|---|
| 论文 PDF (arXiv:2602.24286) | `/data/cuda-agent-migration/paper/cuda-agent.pdf` (4.7 MB, `%PDF-1.7`) | ✅ |
| arXiv 摘要页 | `.../paper/abs_page.html` | ✅ |
| arXiv HTML 全文 (`/html/2602.24286v1`) | 未能下载 | ❌ 服务端 `content-length: 0`，ar5iv 亦返回 "No content available" |
| 项目主页 | `.../paper/project_page.html` (含完整正文与指标) | ✅ |
| 官方仓库 (shallow clone) | `/data/cuda-agent-upstream/` | ✅ 45 files |
| CUDA-Agent-Ops-6K | `/data/cuda-agent-migration/dataset/data.parquet` (1.3 MB, 6000 rows) | ✅ |
| `agent_workdir`（适配副本） | `/data/cuda-agent-migration/agent_workdir/` | ✅ |

**论文元数据**：Weinan Dai, Hanlin Wu, Qiying Yu, Huan-ang Gao, Jiahao Li, Chengquan Jiang,
Weiqiang Lou, Yufan Song, Hongli Yu, Jiaze Chen, Wei-Ying Ma, Ya-Qin Zhang, Jingjing Liu,
Mingxuan Wang, Xin Liu, Hao Zhou（ByteDance Seed / 清华 AIR / SIA-Lab）。Submitted 27 Feb 2026。

**注**：huggingface.co 本机直连不稳定（连接被重置/拒绝），通过 `hf-mirror.com` 镜像完成数据集下载。GitHub 与 arXiv PDF 正常。

---

## 2. 论文要点（适配相关的部分）

三大组件：
1. **数据合成管线**：seed op 来自 `torch`/`transformers`，LLM 组合最多 5 个算子融合成任务，执行级过滤（eager + torch.compile 双模可执行、去随机算子、anti-hacking 去恒定输出、eager 运行时 1ms–100ms、去高相似 KernelBench 样本）。最终 6000 条。
2. **Skill-augmented 环境**：ReAct 式 agent loop + `SKILL.md` 约束 + 编译/验证/profiling 工具。奖励 = 里程碑式离散奖励（正确性 + 相对 torch.compile 的加速）；anti-reward-hacking：受保护的 verify/profile 脚本、禁止 fallback 调用、5 输入正确性检查、同步 warm-up 后 profiling、禁网。
3. **训练**：单回合 PPO warm-up → RFT 初始化 actor + 价值预训练初始化 critic → 多回合 agentic RL。128k 上下文、150 训练回合、评估最多 200 回合。

结果：KernelBench Overall pass 98.8%、faster-than-compile 96.8%、geomean 2.11×；Level-3 94%/90%/1.52×，比 Claude Opus 4.5 / Gemini 3 Pro 高约 40 个百分点。

---

## 3. 环境复用（未改动上游源码）

与 KernelBench 迁移一致，CUDA-Agent 的 **源码未做任何 MACA 专用改动**。兼容性全部由环境层提供：

```
GPU     MetaX C500 — 104 SM, 16.3 GB, warp size 64（非 32）
SDK     MACA 3.3.0.15 (/opt/maca-3.3.0)
编译    torch.utils.cpp_extension → cucc (cu-bridge) → mxcc；无 nvcc
Torch   2.8.0+metax3.3.0.2
```

入口脚本 `/data/cuda-agent-migration/env.sh`（source 后再跑任何命令）：复用 KernelBench 的 env.sh，追加两项——
- `PYTHONPATH` 含 `agent_workdir`（`model_new.py` 需 `import cuda_extension`）
- `LD_LIBRARY_PATH` 含 torch 的 `site-packages/torch/lib`（否则 `import cuda_extension` 报 `libc10.so` not found）

---

## 4. 验证记录

### 4.1 官方示例 axpby 全流程（agent_workdir 内）

| 步骤 | 命令 | 结果 |
|---|---|---|
| 编译 | `bash utils/compile.sh` | ✅ 3 文件，24–25 s → `cuda_extension.so` |
| 正确性 | `python3 -m utils.verification` | ✅ 5/5 (atol=rtol=1e-2) |
| 性能 | `python3 -m utils.profiling` | baseline 6.4 µs / compile 2.8 µs / ext 3.3 µs |

冷启动复现（删 build + .so 后重跑全链路）：编译 ✅、验证 5/5 ✅、profiling ✅，**可复现**。

`sudo` 沙箱形态（SKILL.md 的 canonical 写法）亦验证可用：`sudo -n env ... python3 -m utils.verification` → 5/5 ✅，profiling 同样可跑。

### 4.2 KernelBench 衔接（eval API）

上游 `scripts/run_and_check.py` 依赖 `pydra.REQUIRED`，本机 `pydra` 版本不兼容（与 KernelBench 迁移时相同），直接调 `kernelbench.eval.eval_kernel_against_ref`：

```
compiled: True | correctness: True
metadata: {'hardware': 'MetaX C500', 'device': '0', 'correctness_trials': '(5 / 5)'}
```

**关键衔接点**：`eval_kernel_against_ref` 用 `load_custom_model_with_tempfile` 在隔离上下文加载 `model_new.py`，工作目录不是 `agent_workdir`，因此 **必须** 先把 `agent_workdir` 放到 `PYTHONPATH`，否则 `ModuleNotFoundError: No module named 'cuda_extension'`（compiled=False，correctness=False）。这是 CUDA-Agent 任务格式接入 KernelBench 评估器的唯一真实缺口，由 `env.sh` 解决。

### 4.3 数据集在 C500 上的可执行性（随机 50 行）

- eager 模式可执行 **46/50**
- torch.compile 可执行 **45/50**
- 失败：2× `AttributeError`（样本本身问题）、3× `OutOfMemoryError`

OOM 属 **invalid_infrastructure**（16.3 GB 显存约束），非 backend_error——与 KernelBench L1 静态分析（58/100 可跑、31 OOM）的结论一致。该数据集是为 48 GB 级板卡设计的（运行时窗口 1ms–100ms），在 C500 上需按显存预算筛选任务，不影响正确性闭环。

### 4.4 GPU 常量（SKILL.md 参数调优的输入）

| 项 | C500 | NVIDIA H100 对照 | 影响 |
|---|---|---|---|
| warp size | **64** | 32 | `__shfl_sync` lane 范围、warp 级推理分母；SKILL.md 未提 warp size，需在 MACA 注记中说明 |
| SM 数 | 104 | 132 | wave 数学：`blocks / 104` |
| shared mem/block | 64 KB | 227 KB | SKILL.md 的 tiling 上限更低 |
| 显存 | 16.3 GB | 80 GB | 任务筛选 |

---

## 5. 已发现的适配差异（供后续阶段参考）

1. **示例任务规模在 C500 上无法区分性能**：`model.py` 输入 `1×128`（512 B）。实测 extension 耗时 ≈ 3.3–7.2 µs，而 n=1 的纯 launch overhead 就是 9.4 µs——结果完全由启动开销主导，kernel 本身无区分度。RL 奖励信号若用此规模，噪声极大。**建议**：任务规模按 C500 显存重新定标，使 kernel 计算量显著超过 launch 开销。
2. **`profiling.py` 的 `torch.profiler` device 事件**在 C500 上可工作并给出非零 device_time（profiling 跑通）。
3. **示例 kernel 性能未达 SKILL.md 目标**：ext 3.3 µs vs torch.compile 2.8 µs → 0.85×（比 baseline 慢）。这是上游示例 kernel 的固有水平（naive axpby），非 MACA 问题；也说明 SKILL.md 的 "5% 快于 torch.compile" 门槛在本平台上是真实门槛。
4. **`TORCH_CUDA_ARCH_LIST` 未显式设置**时 PyTorch 会自动选本机可见架构（warning 非错误），与 KernelBench 迁移时一致。SKILL.md 写的 `TORCH_CUDA_ARCH_LIST=9.0`（Hopper）在 C500 上 **不可照搬**，xcore1000 架构不经由该变量选择。
5. **`sudo` 可用**（root 环境），SKILL.md 的沙箱假设成立；但在非 root 的生产 C500 环境需重新确认。
6. **huggingface.co 直连受限**，数据集走 hf-mirror。若后续需自动同步数据集，需固化镜像或代理配置。

---

## 6. 结论

第一阶段（构建闭环 + 正确性闭环）**已打通**：
- 官方示例 compile → verify → profiling 全链路在 C500/MACA 上可跑且可复现；
- `agent_workdir` 与已迁移的 KernelBench 评估器衔接成功（5/5）；
- 训练数据集 92% 可在本机执行，其余为显存容量限制；
- 上游源码零改动，兼容性由 `env.sh` 环境层提供——与 KernelBench 迁移的设计一致。

**下一阶段建议**（按优先级）：
1. 按显存预算筛选/重定标任务规模，消除 launch-overhead 主导的奖励噪声；
2. 用数据集中的真实任务（而非 `1×128` axpby 示例）验证端到端 agent loop；
3. 补 SKILL.md 的 MACA 注记（warp 64、shared 64 KB、无 tensor-core 计数器、profiler 用 mcTracer）。

---

## 附录：Loop 阶段 A–D 成果（2026-09-15）

### 阶段 A：任务筛选与定标

| 项 | 结果 |
|---|---|
| A1 显存预算分级 | AST 静态解析（含模块级常量求值）6000 行；有估值 3454 行：le2GB 2208、le6GB 682、le12GB 278、over16GB 286。2GB 预算容纳 63.9%，4GB→79.0%，8GB→88.5% |
| A2 规模扫描 | axpby 全规模快于 torch.compile；n≥4K 时 kernel 计算开始主导（BW 22→893 GB/s）。发现 `utils/profiling.py` 内联 eager lambda 被 DCE 消除，基线不可信 |
| A3 任务探针 | 10/10 可执行（eager+compile 双模），峰值 1.0–2.8GB，运行时 0.8–13ms，全部落在论文 1ms–100ms 窗口 |

### 阶段 B：axpby kernel 优化

| 版本 | 说明 | vs torch.compile | vs v1 |
|---|---|---|---|
| v1 | 上游朴素版（grid-stride, float） | 1.6–5.6× | — |
| v2 | float4 向量化 + 标量尾数 | **2.44–5.58×** | 1.02–1.59× |
| v3 | + 显式 2-way ILP | 2.31–5.51× | 0.95–1.04×（无增益） |

- B3 KernelBench 官方门槛（cuda_event 计时，`measure_performance=True`）：**1.08× speedup**，correctness 5/5 —— 通过 5% 门槛。
- 最终保留 v2；v1/v3 归档至 `archive/intermediate/`。
- 结论：float4 已足够，编译器自动完成 ILP 调度，手动展开冗余。

### 阶段 C：真实任务端到端

- **C1**（row 2341，ConvTranspose1d+ReLU6+bias）：compile ✅、correctness 5/5 ✅（直接对照 max diff 3.6e-4）。修两个真实 bug：ConvTranspose 索引映射（`w_in=(w_out+pad-k)/stride`，output_padding 不进映射）、`load_state_dict` 参数名必须与 Model 一致。性能 0.36× —— 朴素实现不敌 cuDNN，性能门槛未达成（真实结论，非 bug）。
- **C2**（20 个 ≤6GB 档任务批量验证）：**eager 20/20、torch.compile 20/20**，峰值 0.24–9.46GB（中位 1.13GB），17/20 落在 1–100ms 窗口。

### 阶段 D：文档与收尾

- `SKILL.md` 追加 MACA Supplement（环境、架构常量表、DCE 陷阱、规模下限、实测参考点），上游原文未改。
- `env.sh`（PYTHONPATH + torch lib LD_LIBRARY_PATH）、`LOOP_PLAN.md`、产物 JSON 全部就位。

### 显存使用情况（16.3GB C500）

全程空闲显存维持 15.69–16.16GB，单任务峰值 ≤ 9.46GB，每任务后 `gc + empty_cache`，未对其他 Agent 造成压力。OOM 仅出现在 286 条 over16GB 数据（为 48GB+ 板卡设计），属容量限制。
