# 阶段 3 预注册：精度–意外度前沿（论点 D）和修复后的 SV 臂（2026-10-20，写于任何阶段 3 调用之前）

## 论点
- **D**：在可验证的假设空间上，prompt 和 harness 层面的干预都只是沿着"精度–意外度"前沿移动，推不动它。具体说，没有一个干预能让"为真且低先验"（SV）的数量显著超过直接生成；凡是降低平均先验的干预，都会以精度为代价。
- **A**（已由阶段 2 的 P1 支持）：按 judge 优化会 Goodhart。

## 动机
阶段 2 里 SV 臂一次都没采纳候选，但有两个混杂因素：
1. 不变式正则 `conv\w*` 误拦了 "conventional"，SV 臂第 1 轮 3 个候选里有 2 个被误杀；已修复为 `conv(olution\w*|\d\w*)?`。
2. 起点 BASE 只跑了 6 次，SV 均值 1.5，高于阶段 1 直接生成的 1.25，起点偏高，候选更难超过。

另外，D 需要第二个生成模型来排除"这只是 Sonnet 的特性"。

## 实验
### E1 修复后的 SV 臂（臂名 SV2，生成模型 Sonnet）
- 优化器和阶段 2 完全相同（3 轮 × 3 个候选 × 6 次运行，proposer 是 Opus），只有两处改动：
  - 使用修复后的不变式；
  - 起点 BASE 的分数用 12 次运行估计（rep 0–11，其中前 6 次复用阶段 2 的缓存）。
- proposer 的缓存键使用新的臂名，不复用阶段 2 的提议。
- 最终 harness 在 NB201 上跑全新的 rep 100–107，共 8 次，并由 judge 打分。
- 如果最终 harness 就是 BASE，直接复用 BASE 已有的新运行。

### E2 第二个生成模型：Opus
- 在 NB201 上用 Opus 生成，每个 harness 跑 8 次全新运行（rep 100–107）。不打 judge 分。
- 5 个 harness，全部来自阶段 2，在看到 Opus 结果之前选定：
  1. BASE（直接生成）；
  2. final_J；
  3. final_V；
  4. SV 臂第 2 轮的 "single / pool 20 / prior_avoid" 候选；
  5. SV 臂第 1 轮的 "cross_domain / gap_innovator / flaw" 候选。
- 留一先验 = haiku 和 sonnet 的均值（排除生成模型 Opus）。

## 判据
### D1 推不动
- 对每个生成模型，把每个非 BASE 的 harness 和 BASE 比较 SV（单侧置换检验 5000 次，Holm 校正）。
- 纳入的 harness：
  - **Sonnet**：阶段 1 的 8 个非直接配置，与阶段 1 的直接生成比较；它们是预先指定的，不存在选择偏差。再加上阶段 2 的 final_J、final_V 和 E1 的 final_SV2，用新运行与 BASE 的新运行比较。
  - **Opus**：E2 的 4 个非 BASE harness。
- **支持**：两个生成模型都没有任何比较在 Holm 校正后显著，且所有 harness 的 SV 均值都 ≤ BASE + 0.5。
- **否定**：任何一个比较在 Holm 校正后 p < .05。
- 其余：inconclusive。

### D2 前沿形状
- 对每个生成模型，在各 harness 之间计算 Spearman(平均先验, 精度)。Sonnet 用阶段 1 的 9 个配置，加上阶段 2 的 3 个最终 harness 和 SV2；Opus 用 5 个 harness。
- **支持**：两个模型的 ρ 都 ≥ 0.5，即越离开先验，精度越低。
- **否定**：任何一个模型的 ρ ≤ 0。
- 其余：inconclusive。

### D 的总判定
- D1 和 D2 都支持，D 才算支持。
- 只要 D1 被否定，D 就不成立，并报告是哪个 harness 推动了前沿。

### A 的复核（描述性）
- 在 Opus 生成下，final_J 相对 BASE 的精度和 SV 变化（不打 judge 分）。

## 替代解释
- **NB201 对模型太熟悉，SV 的上限低**：如实承认；阶段 2 的 CTR 迁移测试作为第二个空间（描述性）。
- **3 轮优化不足以找到好的 harness**：如实承认。D 的结论限定在"这个预算下的 harness 搜索"，不推广到权重训练。
- **E2 的 harness 是在 Sonnet 上选出来的，可能不适合 Opus**：这正是测试点。如果 Opus 上有 harness 超过 BASE，D 就被否定。

## 成本（订阅 CLI）
| 项目 | 调用数 |
|---|---|
| E1：BASE 补跑 6 次，3 轮 × 3 候选 × 6 次 × 约 3 次调用，加 proposer | 约 170 |
| E1：最终 harness 新运行 8 次，加 judge | 约 30 |
| E2：5 个 harness × 8 次 × 约 3 次调用 | 约 120 |
| 合计 | 约 320 |
