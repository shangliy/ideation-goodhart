# 阶段 2 预注册：同一个优化器、三种 fitness（2026-10-20，写于任何阶段 2 调用之前）

## 问题
用同一个 harness 优化器、同样的预算，分别以三种目标优化 idea 生成 harness：
- **J**：LLM judge 打分；
- **V**：实测有效性（为真的条数）；
- **SV**：为真且低先验的条数。

问题是：最后得到的 harness 在新运行和没见过的空间上，各自是什么样子？

## 变更说明（相对 07 修订 2 和上一条回复）
- **训练空间仍是 NB201，不换成 CTR。**
  - CTR 表上，Sonnet 直接生成每次运行的 SV 只有 0–1 条（P3：4 次运行一共 0 条），作为训练 fitness 只能看到噪声。
  - NB201 每次运行 SV 均值 0.96、标准差 0.81（阶段 1，72 次运行）。
- **CTR 作为迁移测试**，模型不熟、数据私有。它的 SV 统计效力低，只做描述，外加合并后的超几何检验。
- **token 成本以硬上限的方式进入**：每次运行平均 ≤ 40k token，超过的候选不能被采纳。另外报告成本归一化后的结果。

## Harness（被优化的对象，见 s2.py）
| 字段 | 取值 |
|---|---|
| retrieval | none / in_domain / cross_domain（检索走 search.py，带 NB201 屏蔽名单） |
| roles | single / gap_innovator / gap_innovator_critic |
| challenge | none / flaw / negate |
| pool | 10 / 20 / 30（> 10 时再选出 10 条） |
| select | likely_true / prior_avoid / custom |
| select_rule | ≤ 50 词 |
| strategy | ≤ 150 词，插入生成者（或 innovator）的 prompt |

- **不变式**：文本字段里不允许出现两个空间的专有词（正则 BAN 见 s2.py），违反的候选直接丢弃并记录。
- 角色提示词全部改成领域通用的写法；空间描述（SETUP）和输出格式在 harness 之外，固定不变。
- **起点** BASE = 直接生成（none / single / none / 10，没有 strategy）。三个臂共用这个起点，以及它的 6 次运行。

## 优化器（三个臂完全相同，只有目标不同）
- proposer 是 Opus。它看到：
  - harness 字段说明；
  - 目标的文字描述；
  - 历史记录（每个 harness 的平均分和 token）；
  - 当前最优 harness 的一份输出，不带标签。
- 每轮提出 3 个候选。每个候选用 Sonnet 跑 M = 6 次。
- 采纳规则：候选平均分 > 当前最优的平均分，且 token 不超上限，就采纳。共 3 轮。
- 各臂的分数：
  - J = judge（Opus）给出的新颖度 + 质量，满分 2–20；
  - V = 10 条里为真的条数；
  - SV = 为真且留一先验 < 50 的条数。
  - 留一先验 = haiku 和 opus 的均值（排除生成模型 Sonnet）。
- proposer 只看到本臂的分数，看不到逐条的真值。

## 最终评测（全新运行，rep 100–107，训练时没用过）
- 4 个 harness：BASE、final_J、final_V、final_SV。
- 每个 harness 在 NB201 上跑 8 次，在 CTR 上跑 8 次，每次运行都由 judge 打分。
- 指标：judge 分、真条数、精度、SV、平均留一先验、8 次运行合并后的不同主张数、token。

## 判据（NB201 新运行；两两比较用单侧置换检验，5000 次）
- **P1 judge 的 Goodhart**
  - 支持：final_J 的 judge 分 > BASE（p < .05），且 final_J 的 SV ≤ BASE 的 SV + 0.3。
  - 否定：final_J 的 SV > BASE（p < .05）。
  - 其余：inconclusive。
- **P2 只看有效性导致求稳坍缩**
  - 支持：final_V 的平均先验 ≥ BASE + 3，且不同主张数 ≤ BASE。
  - 否定：final_V 的平均先验 ≤ BASE − 3。
  - 其余：inconclusive。
- **P3 SV 目标**
  - 支持：final_SV 的 SV > BASE，且 > final_V（两者都 p < .10），并且精度 ≥ 0.8 × BASE。
  - 否定：final_SV 的 SV ≤ BASE。
  - 其余：inconclusive。
- **主线成立** = P1 支持且 P3 支持。
- **CTR 迁移（描述性）**：报告同样的指标，以及合并后不同主张中 SV 个数的超几何 p（同 P3 的做法）。不作为判据。

## 替代解释和对照
- **采纳时的赢家诅咒**：最终评测全部用新的 rep。
- **某个臂一次都没采纳**：final = BASE，如实报告，不重跑。
- **NB201 是模型熟悉的空间**，检索可能泄漏：阶段 1 中 294 条结果里屏蔽了 11 条。用 CTR 迁移测试作为检查。
- **proposer 从输出样例里学到领域知识**：不变式只约束 harness 文本。样例里有领域词是允许的，因为 Harness² 的 proposer 本来就看行为样例。所以迁移到 CTR 是关键检查。
- **judge 和 proposer 都是 Opus**，J 臂可能自我偏好。如实报告这一点；另外 judge 只评输出，不评 harness。
- **轮数少（3 轮）**：这是 pilot，不是对优化器的充分评估。

## 成本（订阅 CLI）
| 项目 | 调用数 |
|---|---|
| 训练：每臂 3 轮 × 3 候选 × 6 次 × 平均约 3 次调用，加 proposer | 每臂约 165，三臂约 500 |
| J 臂每次运行的 judge | 约 60 |
| 最终评测：4 harness × 16 次 × 约 3 次调用，加 64 次 judge | 约 250 |
| 合计 | 约 800–850 |
