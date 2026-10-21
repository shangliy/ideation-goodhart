# 阶段 11 预注册：Sonnet × NB201 失败的因果检验（2026-10-21，写于任何阶段 11 调用之前）

## 背景
事后分解（`post_conflict.py`，REPORT 7.2）把 Sonnet × NB201 上"先测量再提出"的失败分成三层：
1. **空间**：NB201 上与先验冲突的证据多半是假象。
2. **呈现方式**：NB201 的 prompt 把测量写成"真实测量"；HPOBench 的 prompt 写明"单次训练种子"。在冲突处提出主张时，两个模型都有 91%（NB201）或约 60%（HPOBench）站在证据一边。
3. **模型**：Sonnet 有 42% 的主张提在冲突处，Opus 只有 15%。

第 2 层与空间完全混在一起，第 3 层是相关证据。本阶段操纵 prompt 来检验这两层。

## 设计
三个臂都**复用 Sonnet 已经设计好的探针**，只改"提出假设"那一次调用：
- NB201 用阶段 5 的探针，rep 117–139，共 23 次。rep 116 的 100 个探针全部解析失败、没有测量值，预先排除。
- HPOBench 用阶段 9 的探针，rep 200–223，共 24 次。测量值按原随机种子重算，与原运行完全相同。

每臂的配对基线，是同一 rep 原来的 A-100 运行：探针相同，只有 prompt 不同。

**各臂（生成模型都是 Sonnet，每次运行 1 次调用）**
- **F，可靠性提示（NB201）**：在 NB201 原 prompt 后补一句真实说明："Each result comes from a single parent cell; the same edit can have a very different effect on other cells, so a single result is weak evidence about how the edit behaves in general."
- **G，先验一致性筛选（NB201）**：原 prompt 不变，在"Propose 10…"一句后加："Propose only hypotheses that the experiments support AND that you also find plausible from your own knowledge of neural architectures; if the experiments and your knowledge disagree about a claim, do not propose it."
- **H，反向框架（HPOBench）**：去掉 HPOBench prompt 中 "trained both once (a single training seed) and"，让说法与 NB201 的写法一致，其余不变。

## 指标
- **主要指标**：精度。
- **次要指标**：
  - SV/次；
  - 冲突主张占比：主张处模型自己的探针证据与它自己的盲先验方向相反；
  - 站在证据一边的比例：在提出的冲突格子上，主张方向与证据一致的比例。

## 判据
三个主要检验一起做 Holm 校正，检验方法是单侧配对 sign-flip 置换检验。
- **F**：
  - 支持：精度 F − A ≥ +0.15，且 Holm p < .05；
  - 否定：差值 90% CI 上界 < +0.05；
  - 其余：inconclusive。
- **G**：判据同 F。另外报告冲突主张占比是否下降至少 0.15（描述性）。
- **H**：
  - 支持：精度 H − A ≤ −0.10，且 Holm p < .05；
  - 否定：差值 90% CI 下界 > −0.05；
  - 其余：inconclusive。

## 预期（事先写下）
- **F 支持**：站在证据一边的比例从 .91 降到 .75 以下，精度回升。
- **G 支持**：冲突主张占比从 .42 降到 .25 以下，精度回升到 ≥ .65。
- **H 不确定**：在 HPOBench 上，冲突证据本来就较常为真（真实效应占比 .56），所以即使更信任证据，精度也未必下降。

**如何解读结果**：
- 只有 F 不成立：呈现方式不是主因。
- G 支持：可以把"先验一致性筛选"加进 ideagen 作为选项。但要同时看 SV，因为它可能以牺牲意外度为代价。

## 替代解释
- **F 和 G 增加了 prompt 长度和"谨慎"语气**：谨慎语气可能普遍降低冒险，而不是专门作用于证据可靠性。为此报告平均先验和 SV：如果精度升高的同时 SV 降到 BASE 以下，就解释为"整体趋于保守"。
- **复用探针**：消除了"设计实验"这一步的差异，但也意味着结论只适用于 Sonnet 自己设计的这些探针。

## 成本
共 71 次 Sonnet 调用（23 + 23 + 25；HPOBench 24 个 rep，失败的 rep 只重试一次），走 Claude 订阅。不需要 judge 调用。
