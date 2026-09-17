# CUDA-Agent → MetaX C500/MACA 迁移路线图

**日期**：2026-09-16
**状态**：组件①②已完成并自测，**阶段 0（G9，LLM 接入 agent loop）已完成**；组件③（RL 训练管线）为下一阶段。

论文（arXiv:2602.24286）的三大组件与我们的迁移状态：

| 组件 | 内容 | 状态 |
|---|---|---|
| ① 数据合成管线 | CUDA-Agent-Ops-6K（6000 条）、任务分级、可用任务池 | ✅ 完成 |
| ② Skill-augmented agent 环境 | agent_workdir + SKILL.md + agent_loop 端到端 | ✅ 完成 |
| ③ RL 训练管线 | PPO warm-up → RFT → 多回合 agentic RL | ❌ 未开始 |

---

## 已完成的基线（后续工作的前提）

- **环境**：cucc/mxcc 工具链（无 nvcc），`torch.utils.cpp_extension` 经 cu-bridge 转译到 xcore1000，编译 23–25s。
- **任务池**：`dataset/actionable_tasks.csv`，3396 条（3156 直接 + 240 降档），G4/G4b 实测验证。
- **11 个真实任务工作区**，3 个通过 KernelBench 官方门槛（selu+tanh 1.36×、log+cos 1.39×、leaky_relu×2 1.58×）。
- **agent_loop.py**：SKILL.md 5 步工作流可程序化驱动，LLM-agnostic（`produce_attempt` 接口），已在 axpby（smoke）与 row311（真实任务，71s/episode）上自测通过。
- **奖励信号已可信**：DCE 修复后 eager/ext 均不再被消除；两个基准（vs eager / vs compile）分开记录。
- **任务分型**（写入 SKILL.md）：内存受限 elementwise 融合 = 甜点；单 SFU 链可赢；双 SFU 依赖链（exp→log）必输；conv/attention = cuDNN 壁垒；bool/int8 输出降权。

---

## 阶段 0：LLM 接入与单任务 rollout（G9）✅ 已完成

**完成时间**：2026-09-16。`agent_workdir/llm_backend.py` + `run_g9.py`，axpby 示例上端到端 episode 通过（模型自主产出三件套 → compile → verification 5/5 → profiling → KernelBench gate `compiled=True, correctness=True`，hardware=MetaX C500）。遗留：axpby 规模太小、奖励无区分度，阶段 1 需在 G1 甜点任务上跑才有统计意义。

---

**为什么先做这个**：组件③ 的任何工作都以"模型能产出可用的 rollout"为前提。如果模型在 SKILL.md 约束下连单个任务都走不通，训练管线没有可收集的数据。这一步成本最低、信息量最大。

**内容**：
1. 在 `agent_loop.py` 的 `produce_attempt` 接口上实现 LLM 后端调用（OpenAI 兼容接口即可，便于切换供应商）。
2. 把 SKILL.md 的约束注入 system prompt：禁止 fallback 调用、禁止联网、5 输入正确性、同步 warm-up 后 profiling、输出必须是 `kernels/*.cu` + `*_binding.cpp` + `model_new.py` 三件套。
3. ReAct 循环：生成 → 编译 → 验证 → profiling → 若失败则把 stderr/verification diff 喂回模型重试（SKILL.md Step 4 要求迭代到正确为止）。
4. 在 5–10 个 G1 甜点任务上跑通，记录：通过率、平均迭代次数、失败模式分类。

**验收**：至少 1 个任务上模型自主完成「编译通过 + 正确性 5/5 + 过 1.05× 门槛」的完整 episode，且失败原因可归类。

**风险**：模型可能反复犯平台特异性错误（`TORCH_CUDA_ARCH_LIST=9.0`、warp 64、shared 64KB）。这些正是 SKILL.md MACA Supplement 要回答的——把失败模式回填进 SKILL.md，形成正循环。

---

## 阶段 1：批量 rollout 与数据收集

**内容**：
1. 把 G9 的单任务循环扩展为批量任务调度器（任务池 → 并发 episode → 结果落盘）。
2. rollout 存储格式：每个 episode 记录完整轨迹（prompt、工具调用、stderr、reward），供 RFT 与 PPO 复用。
3. 显存护栏批量版：每个 episode 启动前查 `mem_get_info`，<4GB 空闲时排队。
4. 在 100–300 个甜点任务上跑出第一批 rollout，统计通过率与 reward 分布。

**产出**：可用于 RFT 的正样本轨迹池。

---

## 阶段 2：训练管线（组件③）

论文的训练分三阶段，我们也按此组织：

### 2a. 单回合 PPO warm-up
- 策略模型在**无工具调用**的设定下学写 CUDA kernel（直接给 model.py，要求输出 kernel 源码）。
- 目的：先建立基础代码生成能力，再进入交互式设定。
- 奖励：编译成功 + 正确性（不含性能，降低早期方差）。

### 2b. RFT 初始化 actor
- 从阶段 1 收集的轨迹中筛正样本（编译通过 + 正确 + 过门槛）。
- **RFT 过滤规则**（论文明确要求）：剔除无效工具调用模式、剔除低效循环（反复重试同一错误），降低 policy collapse 风险。
- 用筛后数据做监督微调。

### 2c. Critic 价值预训练
- 训练 value head 对状态（当前 workspace 状态 + 已执行轨迹）估值。
- 目的：多回合 RL 的 advantage 估计在早期就可靠，避免训练前期的剧烈振荡。

### 2d. 多回合 agentic RL
- 完整 agent loop 作为环境，128K 上下文，每 episode 最多 150 轮交互（评估最多 200 轮）。
- PPO + 里程碑式离散奖励（compile_failed 0.0 / incorrect 0.1 / unmeasured 0.3 / correct_slow 0.5 / pass 1.0）。
- anti-reward-hacking 措施必须与论文一致：受保护的 verify/profile 脚本、禁止 fallback、5 输入正确性、同步 warm-up 后 profiling、禁网。

---

## 阶段 3：评估与对照

- 在 KernelBench 上评估（不是我们自己的任务池），与论文指标对齐：Overall pass rate、faster rate vs eager/compile、geomean speedup。
- 对照基线：torch.compile、论文报告值。
- 消融：SKILL.md 的 MACA Supplement 是否真的提升了通过率（有/无的对照）。

---

## 算力评估

### 本机实测的 C500 参数

| 项目 | C500（实测） | H100（论文参照） |
|---|---|---|
| SM 数 | 104 | 132 |
| 显存 | 16.34 GB | 80 GB |
| warp size | 64 | 32 |
| bf16 峰值吞吐 | 223 TFLOPS @ 8192³ | ~990 TFLOPS（稀疏）/ ~495（稠密） |
| fp32 matmul | 39.5 TFLOPS | ~67 TFLOPS |
| 可见设备数 | **1**（macainfo 显示整机 1 张 C500 + 2 CPU agent） | — |

### 关键瓶颈不是算力，是显存

训练管线里最吃显存的不是策略模型的前向/反向，而是 **rollout 采样**：每个 agent episode 要跑 150 轮交互，每轮可能触发一次 23–25s 的 kernel 编译 + 5 次正确性验证 + profiling。这是**编译密集型**而非计算密集型负载。

- 16.3GB 显存对单个 episode 绰绰有余（G4 实测任务峰值 0.13–8.66GB）。
- 但**并发 episode 数受显存硬限制**：按平均 4GB/episode 计，单卡同时只能跑 ~4 个，还要扣除 PyTorch 编译缓存。
- 实测带宽漂移严重（copy 基线 1350 → 745 GB/s），共享环境下并发采样会互相争用。

### 建议的集群规模

**结论：沐曦 C500 可以用于训练管线，但单机规模不够，且不建议把 C500 当作主力训练卡。**

| 目标 | 建议规模 | 理由 |
|---|---|---|
| **阶段 0–1**（rollout 收集、RFT 数据） | **8–16 张 C500** | rollout 是 IO/编译密集型，吞吐由 episode 并发数决定。按 4GB/episode + 编译缓存，每卡 ~3–4 并发，16 卡 ≈ 50 并发 episode。跑 1000 个任务 × 平均 3 次重试 ≈ 3000 episode，每 episode 71s（实测）+ 编译 25s ≈ 100s，50 并发 ≈ 1.7 小时。这个规模沐曦集群完全可行。 |
| **阶段 2a–2c**（warm-up、RFT、value 预训练） | **32–64 张 C500** | 这些是标准 LM 训练，需要真正的 FLOPS。C500 的 bf16 223 TFLOPS 约为 H100 稠密的 45%。同等 token 量在 C500 上需约 2.2 倍卡数。 |
| **阶段 2d**（多回合 agentic RL） | **64–128 张 C500** | RL 的瓶颈是 rollout 与训练的比值（rollout 通常占 60–80% 时间）。128K 上下文的 KV cache 在 16.3GB 显存上很紧张——单卡几乎放不下 128K 上下文 + 训练优化器状态，必须切分（TP/PP），这进一步降低有效吞吐。 |

### 沐曦算力的可行性判断

**可以用，但有三处需要先验证**：

1. **多卡互联**：本机只有 1 张 C500 可见，无法验证多卡训练。需确认目标集群的互联拓扑（NVLink 级？PCIe？）——多回合 RL 强依赖卡间通信，PCIe 互联会让 TP 效率掉很多。
2. **分布式训练栈**：PyTorch 的 NCCL 在 MACA 上对应的是沐曦自有的集合通信库（`libmccollective` 或类似）。需确认 `torch.distributed` 在 MACA 上可用且稳定，这决定能否直接用 DeepSpeed/FSDP。**这是最大的未知项**，必须先做通信压测试。
3. **128K 长上下文**：论文的 128K 上下文训练在 16.3GB 显存上需要显式 KV cache 管理（PagedAttention 或 ring attention）。沐曦的 vLLM/推理引擎是否支持需单独确认。

**务实建议**：
- **阶段 0–1 用沐曦 C500 集群（8–16 卡）**——rollout 收集是沐曦生态最匹配的负载（编译密集型、单 episode 显存小、无多卡依赖），且能直接验证 MACA 的多卡与分布式栈。
- **阶段 2 的训练部分，若复现论文规模，建议混用或用更高显存的卡**（C500 的 16.3GB 是硬约束，不是优化能解决的）。若无法获得更高显存的卡，则把训练上下文从 128K 降到 32–64K，并接受指标下降。

---

## 优先级建议

```
阶段 0 (G9)  ──→  阶段 1  ──→  阶段 2a/2b/2c  ──→  阶段 2d  ──→  阶段 3
   ↑ 必须先做                                    ↑ 最大投入，需算力确认
   └ 若失败，回填 SKILL.md 后重试（正循环）
```

**阶段 0 是唯一的硬前置**：它同时验证 LLM 能力边界、SKILL.md 是否完备、agent_loop 是否稳定。这三件事都不确定时，投入训练管线的风险极高。
