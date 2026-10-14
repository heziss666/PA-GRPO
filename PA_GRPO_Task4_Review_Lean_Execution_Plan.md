# PA-GRPO 项目精简执行方案（结合 Task 4 实证审核）

> **用途**：供项目主进程、Data Pipeline 线程和 Codex 后续执行。目标是**减少冗余工序，尽快完成真实数据生产、算法对照训练和独立评估**；不是削减项目研究内容。
> **版本**：v1.1｜2026-10-13（项目负责人已确认实施口径）
> **核对基线**：GitHub `heziss666/PA-GRPO`，`main@34ef6f4f0825d042d8ae8f1038afc7daa364a75f`（本次核对时）。Task 4 evidence seal 的历史节点为 `caee9d50f4194d101cc40ee27473b4d9c01a1e25`。
> **性质**：此文是项目负责人已确认的第一版实施口径。后续执行范围、共同 Reward 和对照方法以本文件为准；历史计划与证据保留。尚未执行的实验不视为通过。Parser/Verification 属于 Data Pipeline 分支，主线负责算法接线和统一训练评估。

## 1. 核心决策（一页摘要）

1. **Task 4 直接结项**：认可已封存的真实 H800 GPU Smoke 结果，不重新执行或扩展 Task 4 审计。
2. **MATH + ReClor 数据主线必须保留**：完成解析器修复、真实候选验证、正确/错误配对、AB/BA 构造和可训练 Parquet 导出；不能以现成 Chatbot Arena Parquet 替代最终主实验的数据来源。
3. **只实现研究所需算法**：固定 Global-only、EIS、ALC、PA-Controlled-NoGate 四组训练，Base 只评估。共同使用现有 PA 基础 Reward；PA-Controlled-NoGate 额外使用 Consistency Reward，第一版关闭 sigma gate。
4. **删工序，不删正确性**：删重复全量测试、逐阶段独立 Review、重复 Gate/Evidence 封存、未来扩展接口、过量模型/消融；保留最低限度的数据正确性、身份配对、数值稳定和独立评估检查。
5. **用一个模型、一套数据、一套预算推进**：统一 Judge backbone、固定数据、训练配置和推理评估协议；先完成一个真实闭环，再扩量或补多种子。
6. **不做无效重跑**：解析器修复复用已有 240 条候选；已通过的训练链路复用 Task 4 配置；只在有明确阻塞时修复或重试。

## 2. Task 4 审核结果、可信边界与经验

### 2.1 已确认事实

- **Repository evidence：SEALED；Task 4 overall project gate：PASS。** 用户汇报独立审核无 Critical / Important / Minor 问题，完整测试 `149 passed, 21 skipped`，Evidence 校验 `16/16 OK`，`git diff --check` 通过，敏感信息扫描无发现；以上为 Task 4 封存时的报告，不应写成最新 main 的现时测试结论。
- **真实运行**：`task4_run_003_offload`，成功训练代码 commit `22945d2c187d2a748e94221f3e7834fcbca10d61`；模型 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`；单张 H800，vLLM 同步 Rollout，FSDP2 + LoRA。
- **训练路径得到验证**：8 个成对样本、16 条 AB/BA 训练行，两次优化更新；真实 vLLM 启动；基于 `(pair_id, rollout_slot)` 的配对无缺失/重复，并在 Batch Reorder 后仍正确；PA pair-baseline 指标出现；梯度范数非零且有限；两步 Checkpoint、LoRA 更新及四样本 Controlled Evaluation 完成。
- **运行过关所用关键配置**：

```text
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
actor_rollout_ref.rollout.gpu_memory_utilization=0.50
```

- **历史事故与处理**：Run 002 在保存第二步 Checkpoint 时磁盘耗尽；Run 003 保持相同有效训练配置，修复存储容量后成功。因此正式训练须预估剩余磁盘、控制 Checkpoint 数量，并设置空间下限。
- **重要技术细节**：合并后的 LoRA Adapter 曾需修正配置元数据，使 `r=32`、`lora_alpha=64`、scaling=`2.0`；正式评估时应继续校验最终 Adapter 实际可加载及缩放配置正确。

### 2.2 Task 4 明确**没有**证明的事项

- 只运行两步优化，四样本评估仅检查流程，并**没有**验证 PA、EIS 或 ALC 的准确率提升、泛化能力和统计显著性。
- Task 4 使用现成 Chatbot Arena Pairwise 训练数据，**没有**证明 MATH/ReClor 新数据生产的真实产出质量。
- 完整 Checkpoint、Rollout 树、原始 Reward 日志、已合并 Adapter 权重等未保存在精简 evidence 内，**不能**仅凭 `artifacts/gpu_smoke/task4/` 对原运行重新执行严格验证或恢复训练。
- 成功训练时的代码 commit 不等于后来修正 verifier marker 的 `e38d3a8`，也不等于后来的 main；不能宣称较新代码版本已在 Task 4 当时实际训练。

**处置：Task 4 封存即止。后续训练遇到具体新故障时，只针对故障做最小排查；不得重新要求 Task 4 的完整独立审计或证据包。**

## 3. 当前项目状态及缺口

| 项目 | 依据 / 状态 | 后续处置 |
|---|---|---|
| Task 4 真实 GPU 训练链路 | PASS，已有证据封存 | 复用，不重做 |
| Data Pipeline Phase 1 | `DATA_PIPELINE_SYSTEM_READY`；MATH 7,500、ReClor 4,638；总计 12,138 | 复用现有读取、Split 和导出实现 |
| 真实 40 题候选生成 | 旧 AutoDL 备份存在 30 个 `candidates.jsonl`，总计 240 条（本地曾统计） | 不重新生成；用于 Parser 修复回归 |
| Gold/Candidate Parser v2 | Data Pipeline 已完成；不重新开发 Parser | 数据分支复验同一批 240 条候选 |
| Verification 标签问题 | 数据分支仍在处理；最新 smoke 结果尚未通过 | 数据分支解决误标并确认有效标签 |
| 正负 Pair / AB/BA / Parquet | 现有生产代码已实现；smoke 记录为 9 个机器 Pair、8 个审查可信 Pair，正式训练数据尚未交付 | 有效标签确认后由数据分支输出真实训练 Parquet |
| MATH/ReClor 扩量生成与正式训练数据 | 尚未确认已经完成 | 必须完成 |
| Global Advantage / Explicit Identity | Global 纯函数已有，Explicit Identity 已由 Task 4 验证 | 主线直接复用并接入统一训练模式 |
| Local / EIS / ALC / PA-Controlled-NoGate | 主线新增统一 scalar Advantage 接线 | 少量公式/身份/数值测试后执行短 GPU 检查 |
| 四种方法真实训练、独立评估 | 尚未看到完成证据 | 最终核心交付 |

**注意**：历史测试与统计来自各自对应版本和阶段；未经重新检查，不把它们写成当前全项目统一 PASS。

## 4. 必须保留的最短数据生产链

```text
MATH / ReClor（已有原始数据、规范化、去重、90/10 Split）
    ↓
Smoke：复用旧 240 条模型候选
    ↓
使用已有 Gold Parser v2 / Candidate Parser v2 复验（数据分支负责）
    ↓
math-verify（MATH）与 exact label match（ReClor）
    ↓
按来源、模型、难度统计 correct / incorrect / invalid 和可配对率
    ↓
扩量生成（仅对缺少正/负候选的题目补采样；记录配置与随机种子）
    ↓
每个训练题目选择固定的 correct + incorrect Response Pair
    ↓
构造 AB / BA 双顺序，保持 pair_id、permutation_id、rollout_slot 身份规则
    ↓
生成 verl 可直接读取的 Judge Pairwise Parquet
```

### 4.1 Parser v2 后的复验：由 Data Pipeline 负责

Gold/Candidate Parser v2 已实现。直接复验原有 240 条候选，原始 Response 和 Candidate ID 保持不变。主线不重复开发数据 Parser；剩余 Verification 标签问题由数据分支解决。

- **Gold Parser**：官方 MATH `solution` 中取最后一个完整、括号平衡的 `\boxed{...}`；ReClor 使用官方 `gold_label`。
- **Candidate Parser**：允许经明确确认、无歧义的结尾标点或 LaTeX 结束定界符；仍拒绝缺失答案、重复或冲突的最终答案、无法闭合的 boxed、被截断的无效结果。不能为了提升正确率而任意提取最后一个答案。
- **数学判定**：MATH 继续使用固定版本 `math-verify` 的 `parse + verify`，且保留超时与异常处理；`invalid` 不得自动当作 `incorrect`。
- **回归数据**：优先对历史 240 条候选重新验证，抽查新旧状态发生变化的记录，分类记录真正数学错误、格式误判、生成截断和解析异常。
- **结束条件**：未发现新的系统性误标；能从真实候选中可靠抽出正负 Pair；必要时只补充覆盖已发现 bug 的小型单元测试。

### 4.2 扩量生成的最短策略

- 以**有效 Judge Pairs** 而不是候选回答数为目标。建议先达到 **500～1,000 个可用 Pair** 作为首批真实训练集，数量随正确率和生成成本调整；该数是建议目标，不是预先确定的真实产出。
- 保留原计划的三生成模型来源（Qwen2.5-7B、Llama-3.1-8B、Qwen2.5-32B），但**不强迫对已经产生有效正负样本的题目重复生成**；能通过追加/续跑取得缺失答案时，不全量重跑。
- 每题一个固定训练 Pair（来自该题正确/错误回答），并构造两种顺序 AB/BA；任一题缺少正或负样本则暂不进入训练。
- 扩量时保留来源、类别、难度和产出率统计，避免数据全被高产出、容易题占据。
- 生成 Prompt 不得携带 `solution` 或 `gold_label`；训练数据和独立评估题目必须按原始题目身份隔离。
- 保留生成任务 ID、模型版本、Seed、候选 ID 与最小运行记录；**不新增**另一层重复的 Manifest/Gate/跨平台哈希系统。

### 4.3 数据生产最低验收（一次即可）

1. 抽样核对正负标签与模型 Response 对应，不包含批量系统性错标。
2. 每个 Pair 只有同一原始题目的正、负两条回答；AB/BA 标签镜像正确；原始 ID、`pair_id` 一致。
3. Train 与独立评估集无原始题目 ID 交叉；真实 Parquet 能被当前 verl 数据加载入口正确读取。
4. 输出 Pair 数、各来源数量、无效率、补采样次数和输入配置。**不需要**另起一轮逐阶段独立审计、逐项封存或几十条 Gate 审批。

## 5. 算法与训练：只做真正需要的新代码

### 5.1 保留的算法

| 方法 | 训练？ | 目的 |
|---|---|---|
| Base | 否，只评估 | 未训练的性能起点 |
| Global-only | 是 | 跨 AB/BA 的 Global Advantage 基线 |
| EIS | 是 | 固定 `A = A_g + A_l`，验证 Local 的贡献 |
| PA-Controlled-NoGate | 是 | 共同 Base + Consistency Reward，response-level Global Advantage，关闭 sigma gate；整体方法对照 |
| ALC | 是 | 本项目改进：自适应 Local Correction |

ALC 第一版固定：

\[
A_{\mathrm{ALC}}=A_g+\lambda_{\mathrm{local}}A_l,\qquad
\lambda_{\mathrm{local}}=\operatorname{clip}\left(\frac{|\mu_{AB}-\mu_{BA}|}{\sigma_G+\epsilon},0,1\right).
\]

**共同 Base Reward 已确定**：直接复用 `my_reward/judge_qwen.py` 的 `r_base = r_correct + r_format + r_length`，correctness 为 `+1/-1`，format 为 `+0.3/-0.3`，length 沿用现有分段计算；不另写 `unified_j4r`。四组使用相同基础 Reward。训练 Judge 的正确性判断复用既有 controlled A/B answer parser，与 Task 3 统一评估相容；MATH/ReClor 的候选 Parser 由数据分支维护。

| 训练模式 | Reward | Advantage | Sigma gate |
|---|---|---|---|
| `global_only` | Base | `A_g` | 关闭 |
| `eis` | Base | `A_g + A_l` | 关闭 |
| `alc` | Base | `A_g + lambda_local * A_l` | 关闭 |
| `pa_controlled`（PA-Controlled-NoGate） | Base + Consistency（系数 1.0） | Global | 关闭：threshold=null，source=unspecified |

`A_g` 按 `pair_id` 跨 AB/BA 计算；`A_l` 按 `(pair_id, permutation_id)` 计算。使用 response-level scalar reward、population std（correction=0）、`epsilon=1e-6`，计算后广播到 response mask。ALC 的 imbalance 和权重仅由 Base Reward 计算。完整 `(pair_id, rollout_slot)` 配对是运行条件；身份缺失或重复直接报错。

**PA-Controlled-NoGate 已最终确认**：PA v2 §3.4 Eq.(12) 定义低方差门控，但 Appendix F 及官方配置未给出可信 delta。第一版正式对照主动关闭 gate，配置为 `sigma_gate_enabled=false`、`sigma_gate_threshold=null`、`sigma_gate_source=unspecified`；沿用 `(r - mu_G)/(sigma_G + epsilon)` 和 Consistency Reward。`epsilon=1e-6` 和官方兼容路径的 variance floor 均不是 delta。不为 delta 新增调参或审批。该对照不能标为完整 PA-Paper 复现；以后若研究 gate，使用显式有来源的实验超参数并独立命名。

**公平比较边界**：Global-only / EIS / ALC 使用相同 Base、rollout budget、数据及 KL/优化设置，用于分析 Local/Adaptive Local 的贡献。PA-Controlled-NoGate 为**整体方法对照**；不将其差异仅归因于 Advantage。PA-Official 保留用于既有 reproduction，不进入第一版主表。ALC 暂不追加 Consistency 变体。记录 Global/Local/最终 Advantage 的 std 或 RMS，识别更新尺度差异。

### 5.2 最小算法开发任务

1. 利用现有 `permstudy/advantages.py` 和 Task 4 显式身份接线，实现 Local/EIS/ALC 与 PA-Controlled-NoGate，通过 `permutation_training.mode` 统一选择模式；门控第一版关闭。
2. 在既有 `verl/trainer/ppo/ray_trainer.py` 的 Advantage 路径最小接入；**不重写 vLLM、Trainer、RewardManager、数据加载器**。
3. 只保留少量关键数值测试：子组映射正确、Global/Local/ALC 公式、零/低方差稳定性、mask 形状/无 NaN、Batch Reorder 后身份不丢失。
4. 先用已有 Pairwise 数据做 **1 次短 GPU 接线检查**：Reward/Advantage 有限、无 fallback、梯度非零、Adapter 能保存加载；正式实验切换为数据分支交付的 MATH/ReClor Parquet。不重做 Task 4 Review/Evidence Seal。

### 5.3 训练策略

- **Judge backbone 已确定为 Qwen3-8B**（项目负责人确认）。四组和 Base 使用 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，复用 Task 4 已验证环境；各方法不得换模型或 revision。
- 先从 **200 Pair Pilot** 测吞吐量、显存和有无训练信号，再扩展到 500～1,000 有效 Pair；如数据/预算允许再扩大。Pilot 的目的不是单独制作新 Evidence，而是快速排除阻塞。
- 四组方法统一数据、Seed、Batch/LoRA、生成温度、Rollout 数、步数/Token Budget 和评估协议；初版一个训练 Seed，保留最小可复现记录。
- 保留 Task 4 已知有用的 CPU Offload 和显存参数，视新数据长度做必要调整。训练前检查磁盘空间；训练时只保留故障恢复所需最近 Checkpoint，最终保留实际可加载的 LoRA Adapter（及必要 config）。
- 不进行 1.5B/3B/7B/8B 多阶段全套重复训练，不做广泛超参搜索，不按每个 Method 新建完整 Task 4 证据链。

### 5.4 直接运行口径

统一开关为 `permutation_training.mode=global_only|eis|alc|pa_controlled`，`pa_controlled` 在本轮始终标记为 PA-Controlled-NoGate。未指定该开关的旧命令仍走原 official 路径。

在现有 Linux GPU 环境使用 `scripts/run_controlled_pairwise.sh`；`TRAIN_DATASET` 必须含完整且相邻的 AB/BA 行。短接线检查默认 2 steps、rollout.n=2、batch=16，正式训练按已统一的预算设置 `STEPS`、`TRAIN_BATCH_SIZE` 等。保持 `data.shuffle=false`；逐行随机打散会把同一题的 AB/BA 分到不同 batch，因此不适用于当前最小接线。

```bash
METHOD=alc \
MODEL_PATH=/private/path/to/pinned-Qwen3-8B \
TRAIN_DATASET=/private/path/to/complete-pairwise.parquet \
RUN_DIR=/private/path/to/new-alc-run \
bash scripts/run_controlled_pairwise.sh
```

只保留最终一次 checkpoint，输出普通训练日志。最终 Adapter 使用既有流程导出并校验：`python -m verl.model_merger merge --backend fsdp --local_dir "$RUN_DIR/checkpoints/global_step_2/actor" --target_dir "$RUN_DIR/final-hf"`（步数随 `STEPS` 变更），然后运行 `python scripts_permstudy/task4/normalize_merged_adapter.py --adapter-dir "$RUN_DIR/final-hf/lora_adapter" --expected-rank 32 --expected-alpha 64`；加载/受控评估复用现有 evaluator，全部模型产物放在私有目录。

接线完成不等于真实 GPU 训练完成；必须有实际优化、保存/加载及评估结果后才能更新该状态。

## 6. 评估与结果：最低但可解释

- **评估数据**：使用从训练题目身份隔离的 Holdout 或独立 Judge Benchmark；若使用 MATH/ReClor Holdout，需要以评估专用方式构造 Judge Pair，不能从训练侧泄漏回答、候选或 gold 信息。
- **统一推理配置**：Base、Global-only、EIS、PA、ALC 使用同一选项格式和解码设置；对 AB/BA 两种顺序均评估。
- **核心指标**：Accuracy、Semantic Consistency、Consistent Accuracy（CA）；报告有效输出比例/格式无效率，避免不同算法格式差异造成误导。
- **训练诊断**：Reward、Policy Loss、Gradient Norm，以及至少一个与 Pair/Local Correction 有关的统计量（例如 `|μ_AB−μ_BA|`、`λ_local`）。
- **低成本机制分析**：对按 subgroup imbalance 分组的样本做离线统计，解释 Global-only、EIS 和 ALC 的差异；尽量复用训练时的 Reward/Advantage 记录，而不是再开一批训练实验。
- **结论边界**：单种子、较小数据集只能支持初步趋势，不能宣称统计显著优越；ALC 没有胜出也要如实记录。

最终至少交付：**一张方法对比表、一张关键机制图、一份简洁 README、可复现的配置/命令、最终模型 Adapter（私有存放）和对应评估摘要**。

## 7. 本次明确砍掉 / 延后的内容

| 类别 | 砍掉或暂停 | 保留的最小替代方案 |
|---|---|---|
| Task 4 工程审核 | 再跑 Task 4、再封存 evidence、再做完整独立审计 | 使用现有 `SEALED/PASS` 结果；新错误定点修复 |
| 数据质量流程 | 每阶段独立审批、逐层人工 Gate、全量人工审计 | 一次核心质量检查 + 异常抽查 |
| 生成流程 | 解析器改动后全量重新生成已有候选；重复产出已经有效的题目 | 复用 240 条旧候选；按缺口补采样 |
| 数据基础设施 | 新的通用多后端框架、多重 Manifest/Hash 体系、未来扩展 Schema | 直接复用 Phase 1 现成实现和必要身份字段 |
| 重复单测 | 每次微调都跑全量测试、跨环境重复相同测试 | 改动相关测试先跑；最终交付前再跑全量一次 |
| 算法扩展 | ALC+Consistency、多套 ALC 权重公式、独立 Gate 阈值网格 | 单一 ALC 公式 + Global/EIS/PA 必要对照 |
| 训练矩阵 | 1.5B 七法完整消融、多模型尺寸全流程、多 Seed 全量重复 | 固定一种 Judge，四组训练 + Base 评估，首轮单 Seed |
| 额外研究问题 | MCQ P>2 扩展、Sensitivity Filtering 等规模多随机抽样消融 | 本轮不进入主线，留作后续研究 |
| GPU 日志/存储 | 每步完整 Checkpoint、所有 rollout 长期归档、重复生成巨型审计包 | 最小日志与可恢复 checkpoint + 最终 Adapter |
| 项目组织 | 每项单独 proposal、分支/工作树/审查循环、十几项子任务流水账 | 一个短执行清单；只针对真实阻塞发任务 |

**不可删**：数据标签真实性、Train/Holdout 隔离、正确数学验证、正负 Pair 真实性、AB/BA 语义和显式身份、关键数值稳定性检查、最终可加载 Adapter、独立评估。它们不是“过度工程”，而是研究结论成立的前提。

## 8. 最短执行顺序、负责人交付与停止规则

| 顺序 | 任务 | 交付物 | 继续推进的条件 |
|---|---|---|---|
| 0 | **冻结 Task 4**；同步当前 main，保留历史 evidence | 固定训练代码基线与 Task 4 配置 | 不要求重新审核 |
| 1 | 数据分支用既有 Parser v2 复验旧 240 条候选并解决标签问题 | 新 Verification 结果和错误分类/Pair 产出率 | 标签可信 |
| 2 | 扩量生成 MATH/ReClor、构造 Judge Pair 和 AB/BA、导出 Parquet | 一批真实可训练 Pairwise Parquet | 标签/身份/隔离检查通过 |
| 3 | 主线统一 PA Base Reward，接入 Global/Local/EIS/ALC/PA-Controlled-NoGate | Trainer 模式开关与少量数值测试 | 公式正确、身份正确、无静默回退；PA gate 关闭 |
| 4 | 一次短 GPU Pilot，再跑四组正式训练 | 四组最终 Adapter、最小训练日志 | 真实训练完成、关键数值有限 |
| 5 | 统一独立评估、离线机制分析、结果整理 | 表格、图、README 和复现命令 | 不夸大效果、可对应实际记录 |

阶段 **1/2（数据）** 与 **3（算法）** 可以有限并行，但须在启动正式训练前统一训练数据 schema、Trainer 输入与选定代码 commit。不要多个线程同时修改同一训练核心；不要为了“并行”引入额外集成成本。

职责固定：Data Pipeline 线程维护 Parser/Verification/补采样/Pair/ABBA/Parquet；主线维护共同 Judge Reward、Advantage 和训练评估。只通过已有 schema 与 Parquet 接口汇合。不得增加新审批层、重复 Evidence Seal 或逐阶段独立 Review；测试集中于新公式、AB/BA 身份、真实模型更新与正式数据标签/隔离。

### 8.1 真正需要停下来检查的情况

- MATH/ReClor 解析器出现**系统性假正确**，或无法确定标注可靠性；
- 生成结果长期无法构成有效正负 Pair，且补采样未改善；
- Train/评估样本身份重叠或 AB/BA 语义映射错误；
- 新 Advantage 存在 NaN/Inf、配对错乱、训练代码静默退回其他算法；
- 训练 OOM、磁盘不足或最终 Adapter 无法加载。

除上述真实阻塞外，**不再把额外审核或新功能当作阶段前置条件**。修复后继续主线，不重启整套 Task 计划。

## 9. 项目最终“完成”的定义

只有同时取得以下**真实交付**，才能称项目第一版完成：

1. 使用 MATH + ReClor 的**真实模型候选**构造的可验证 correct/incorrect Judge Pairwise 训练数据；有清楚的来源、数量、隔离和有效率统计。
2. Global-only、EIS、PA-GRPO、ALC 四种方法在**同一 Judge backbone 和训练数据**上真正完成训练，各有可加载的最终 Adapter 或等效模型产物。
3. Base 加上述四种方法在同一独立评估集上得到 Accuracy、Consistency、CA（及格式有效率）的真实比较结果。
4. 至少一项 ALC 机制分析（如 imbalance bins 与 local correction 权重）能够用真实训练数据或 rollout 统计支撑。
5. README 交代算法、关键配置、可复现命令、结果、限制；大模型权重、完整私有题目、候选文本和敏感路径**不提交 GitHub**。

如果截至截止时间仅完成短 GPU Smoke，则应标注为**工程闭环验证完成、研究实验未完成**；如果只有单种子且数据偏小，则标注为**初步对照结果**，不宣称显著性。

## 10. 给主进程 / Codex 的执行指令（可直接引用）

> 执行已确认的最小科研闭环。Task 4 结项；Parser v2 不重复开发，原 240 条候选不重新生成。Data Pipeline 负责复验、解决标签问题、扩量与真实 MATH/ReClor Parquet。主线复用 PA 基础 Reward 和既有 controlled Judge parser，接入 Global-only/EIS/ALC/PA-Controlled-NoGate；PA 添加 Consistency，四组均关闭 sigma gate，delta 保持 null/unspecified。不为 delta 新增实验。统一 Qwen3-8B backbone、数据、预算和评估，先用已有 Pairwise 数据检查接线，随后跑正式四组与 Base 评估。只新增公式/身份/数值/更新/数据隔离所需检查，不新增完整 Review/Evidence/Gate 链。真实阻塞定点修复；尚未执行的实验不能写成完成。

---

## 参考依据

- [Task 4 evidence README](https://github.com/heziss666/PA-GRPO/blob/main/artifacts/gpu_smoke/task4/README.md)
- [Task 4 真实 GPU Smoke 执行计划](https://github.com/heziss666/PA-GRPO/blob/main/Codex_Task4_Real_vLLM_GRPO_Smoke.md)
- [Task 4 `metrics_summary.json`](https://github.com/heziss666/PA-GRPO/blob/main/artifacts/gpu_smoke/task4/metrics_summary.json)
- [Phase 1 数据流水线结果](https://github.com/heziss666/PA-GRPO/blob/main/artifacts/data_pipeline/phase1/summary.json)
- [Controlled Training Data Pipeline 设计](https://github.com/heziss666/PA-GRPO/blob/main/docs/superpowers/specs/2026-10-06-controlled-training-data-pipeline-design.md)
- [现有算法计划 `PLAN.md`](https://github.com/heziss666/PA-GRPO/blob/main/PLAN.md)
- [Pairwise/MCQ 边界说明](https://github.com/heziss666/PA-GRPO/blob/main/PLAN_ADDENDUM_PAIRWISE_SCOPE_AND_MCQ_EXTENSION.md)

> 此方案按当前对话中的实际目标缩减后续执行范围。它不会更改、覆盖或者删除 GitHub 历史计划与 Task 4 证据；相关的正式设计变更应作为明确的项目范围决策记录，不得将尚未执行的任务写成已验收。
