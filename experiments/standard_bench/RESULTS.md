# 小规模标准 benchmark：原版 BoltzGen vs 当前 Omni

固定使用同一个 final.pt（1000 步），没有新增训练；每题采用相同输入、生成种子及采样步数。

| 能力 / 标准指标 | 原版 | Omni | 差值（Omni − 原版） | 配对 bootstrap 95% 区间 |
|---|---:|---:|---:|---:|
| CASP16，10 题，Cα lDDT ↑ | 0.5259 | 0.5130 | -0.0129 | [-0.0323, +0.0072] |
| CASP16，10 题，TM-score ↑ | 0.5229 | 0.5159 | -0.0070 | [-0.0348, +0.0222] |
| MotifBench，6 题 × 2 候选，成功率 ↑ | 0.4167 | 0.0000 | -0.4167 | [-0.6667, -0.1667] |
| 补充：每个候选最佳 scRMSD 均值，Å ↓ | 1.1362 | 121.9072 | +120.7711 | [+65.9626, +174.9985] |

Design 成功数：原版 5/12，Omni 0/12。

Design 使用官方 MotifBench ProteinMPNN（每个 scaffold 8 条序列）+ ESMFold，成功要求同一条回折结果同时满足 scRMSD < 2 Å、motif RMSD < 1 Å。没有另加身份一致性过滤。

补充的连续读数直接来自官方 scRMSD 输出：每个候选取 8 条评估序列中最小的 scRMSD，再平均。它描述与设计骨架偏离多大，不替代同时满足两个 RMSD 阈值的成功率主指标。

Folding 的 lDDT 仅计算实验结构中已解析的 Cα，采用 Biotite 标准实现；TM-score 采用 TM-align。两者均为 0–1 标度。这不是完整 CASP 官方排名。

抽样在预测前锁定：CASP16 按长度取 3/4/3 题并去掉适配训练/验证的序列同源命中；MotifBench 三个片段数分组各取短、长两个不同 PDB 的任务。完整题目和逐题结果见同目录 CSV 与 manifest.json。

解读：只有 folding 和 design 各自提升，才能支持“双任务都更强”。区间跨零则该小样本不足以下稳定结论；两边全零时 bootstrap 零区间不表示能力相同，而表示该测试未分辨出成功率差异。设计区间按 6 个任务重采样，不把 8 条配序列当独立题目。

范围：MotifBench 检验 scaffold 设计能力，序列经 ProteinMPNN 重设计，因此不能单凭它证明 Omni 自身序列头更好。原版预训练是否见过这些公开目标未排除，结果是配对回溯比较，不是盲测泛化证明。

来源：[CASP16 公共目标](https://predictioncenter.org/download_area/CASP16/)；[MotifBench 官方代码与定义](https://github.com/blt2114/MotifBench)。

本次标准测试累计分配 GPU 0.523 小时（含中断后续跑），没有新增训练。时间来自 Slurm 实际作业记录，排队时间不计入。
