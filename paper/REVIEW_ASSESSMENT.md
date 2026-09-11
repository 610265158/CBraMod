# 模拟 ICLR 审稿意见（基于当前 32 页版本）

## 总体结论

**可以投**。论文完整、自洽、诚实限定、实验量在投稿论文中属于中上水平。预测分数落在 **borderline 区间（均分约 5–6）**，两极风险明显：遇到重视"严格 baseline 评测"的审稿人可能给 6–7，遇到重视"方法新颖性"的可能给 3–4（"这就是个 reshape"）。有两个补实验可以显著把分数往上拉（见下）。

---

## Summary（审稿人会这样概括你的论文）

> 本文提出 lossless phase-interleaved temporal folding：把 [C,T] 的 EEG 无损重排为 (CP)×(T/P) 的"图像"，使 ImageNet/DINOv3 预训练的视觉 backbone 可以直接处理 EEG。在 12 个基准上与已发表的 supervised 模型和 EEG foundation models（CBraMod、REVE）对比，4M 参数的 EfficientNet-B0 在多个任务上超过 69M 的 REVE-Base；随机初始化对照证明视觉预训练本身贡献显著（κ +7.5~27.8）。

## Strengths（审稿人会认可的）

1. **实验严谨度高于领域平均**：12 数据集 × 5 seeds、validation-only 选点、每 seed 只评一次 test、population std、协议在附录可复现。
2. **Split 可核验**：12 个数据集全部继承 CBraMod/REVE 开源代码的固定划分（8 共有 + 3 CBraMod-only + HMC←REVE），TUEV 版本差异处理透明（官方 eval 集、29,421 段不变）。
3. **随机初始化对照是最有力的证据**：6 组配对比较、同一 recipe 只换初始化，说明"视觉预训练本身可迁移"，而不只是"架构能拟合"。
4. **参数/FLOPs 表**：B0 4.01M/0.25G vs CBraMod 4.92M/0.79G vs REVE 69.19M/11.07G——"compact"主张可量化、可核验（含 CBraMod 论文 4.0M 与 release 4.92M 差异的诚实标注）。
5. **负结果诚实**：TUAB/SEED-V/Mumtaz 落后于最强基线、FACED 架构敏感、conditional free lunch 的定位，降低 overclaim 风险。
6. **Threats to validity 完整**，恰好覆盖四类高概率质疑。

## Major Weaknesses（审稿人会攻击的，按严重度）

1. **Novelty：'这只是一个 reshape'**（最大风险）。
   论文的防御是双射性证明、感受野分析、初始化对照——但**缺少 permutation control**（phase-shuffle / channel-shuffle 对照）。§6.2 自己写了"should be tested with folding and permutation controls"。如果 phase 打乱后性能崩溃、而正常折叠有效，"布局技巧有效"的因果链就闭环了；没有这个实验，审稿人可以说"收益可能来自任何 2D 化/加高输入，与 phase 结构无关"。
2. **Baseline 是转录数字，非同管线复现**。
   CBraMod 官方 README 自己承认 BIOT/LaBraM 的 split 不能保证一致、其对比"may not be entirely fair"。你们的威胁小节已限定"CBraMod/REVE rows 同 split、其余为转写参考"，但审稿人仍可能要求至少在 1 个数据集上用官方代码同管线复现 CBraMod。
3. **随机初始化对照只覆盖 2/12 数据集**，且 random-init 的 B0（κ=.5995）已高于已发表的 BIOT（.5273）——会被问"是不是任何认真训练的模型都够"。
4. **REVE 对比的 scope**：REVE 主打 linear probing/few-shot/cross-montage；full fine-tuning 下 69M/11.1G 输给 4M/0.25G 并不出人意料。论文已限定 scope，但建议在正文明确"REVE 的设计目标是 LP/few-shot"以免被认为树靶子。
5. **逐数据集调参**：已承认 + prespecified geometry rule，可接受。

## Minor Issues

- 主文 **~11.2 页 > 9 页限制**（硬性 blocker，必须再砍 ~2 页；方案已列：locality 推导入附录、Table 1 瘦身、§5.5 细节入附录等）
- 尚未用**官方 ICLR 模板**编译验证（fallback 版式略宽松，真实页数可能更多）
- 刚已修正：CBraMod 引用更新为 ICLR 2025、REVE 更新为 NeurIPS 2025（原为 arXiv preprint）；binary 表注的 "All methods use the same partitions" 改为精确表述（BIOT/LaBraM 行是转写参考）
- 附录可视化图的文字在缩放后偏小（side-top-view 等）
- 建议提供**匿名代码仓库**（ICLR 不强制，但对本篇"可核验性"是重要加分）；注意脚本里有本机绝对路径，发布前要脱敏

---

## 提升接收率的两个补实验（按性价比）

1. **Phase-shuffle / channel-shuffle 对照**（在 TUEV ± 1 个数据集上即可，训练成本低）：
   - phase-shuffle（打乱 p 顺序，内容不变）→ 若性能大幅下降，证明"phase 结构"是收益来源
   - channel-shuffle（打乱电极顺序）→ 测试行序敏感性
   这直接封堵"just a reshape"攻击，是论文逻辑闭环的最后一块。
2. **同管线复现 CBraMod**（至少 TUEV 或 TUAB 一个数据集）：官方代码 + 你们相同的 split/seed/选择协议。若复现值与发表值接近，"baseline 没调好"的质疑消失；若更高，也是诚实报告。

## 投稿前 Checklist（按优先级）

- [ ] **必做**：主文压到 9 页内（还差 ~2 页）+ 官方 `iclr2027_conference.sty` 编译验证
- [ ] **必做**：通读一遍终稿（可交给我做 copyedit 轮）
- [ ] **强烈建议**：补 phase-shuffle 对照（1–2 个数据集 × 3 backbone × 5 seeds，约几十 GPU 时）
- [ ] **强烈建议**：单数据集同管线 CBraMod 复现
- [ ] 建议：匿名代码仓库（路径脱敏）
- [ ] 已完成 ✅：随机初始化对照、参数/FLOPs 表、HMC 补 REVE 对照、Mumtaz/HMC/TUEV 事实修正、threats 精简、引用 venue 更新

---

**一句话**：论文现在是一个"诚实、可核验、实验量扎实"的 borderline 提交；补上 permutation control 和一个同管线 CBraMod 复现，就有实打实的 borderline-accept / accept 实力。要我先执行哪一项？（压缩到 9 页我可以直接做；permutation 实验我可以把配置和启动脚本准备好。）
