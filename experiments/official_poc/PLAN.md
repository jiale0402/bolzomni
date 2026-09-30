# 官方数据小规模 PoC

目标：一个共享 Omni checkpoint 的 folding 和 design 能力均优于原版 BoltzGen。
只训练 Omni，从原始 boltzgen1_diverse.ckpt 重启，不使用旧失败权重，不训练额外对照。

| 元素 | 固定安排 |
|---|---|
| 训练数据 | boltzgen/boltzgen1_train 官方 targets.zip 的真实 NPZ；128 条 80–160 aa 完整单体，另留 16 条验证；不同官方序列簇；发布日期不晚于 2023-06-01；保留成员 SHA256、CRC、官方记录 |
| 模型 | 冻结 trunk；diffusion、conditioning bridge、反馈模块、序列头全参数更新；无 LoRA |
| 修复 | 恢复原生坐标旋转/平移增强和 smooth-lDDT；训练与采样都只固定 motif 骨架；坐标、加噪、锚点使用同一坐标系；同一 optimizer update 累积 4 个 folding + 4 个 design 样本 |
| 损失 | sigma 加权坐标 MSE + 原生 smooth-lDDT + masked identity CE + 双任务原版教师 MSE；CE 系数按更新前实测的全参数与共享 diffusion 梯度比例确定，并冻结 |
| 更新 | 400 次，effective batch=8；diffusion 最大 LR 5e-6，新模块 1e-4；40 步 warmup；AdamW β=(0.9,0.95)；原生 norm clip=10；固定最终第 400 步 checkpoint |
| Folding | 作者公开 val_monomers_boltzgen_min50_max220.txt 中 8 条；每条每模型 5 次同种子采样；best-of-5 Cα-lDDT、TM-score，另报全原子 lDDT 与 5 次平均分 |
| Design | 论文 Table 8：6E6R-short、1YCR、1PRW、5YUI、5IUS、1BCF；覆盖 1–4 个 motif 片段；每题每模型 4 个固定候选；每个骨架 8 条 ProteinMPNN 序列，用 ESMFold 重折叠 |
| Design 成功 | 同一条重设计序列同时达到 Cα-RMSD≤2Å、motif backbone RMSD≤1Å、pLDDT≥70、平均 pAE≤5；全部候选计入分母；另报 TM≥0.6 单链接去重成功数 |
| 公平性 | 同一输入、种子、500 diffusion steps、3 recycles；原版不训练；两任务只用一个 Omni checkpoint；不根据测试结果挑 checkpoint、题目或种子 |
| 记录 | W&B + JSONL，数据/源码/权重 hash、各 loss、加权梯度、裁剪前后 norm、固定 train/validation probe、采样几何、最终权重、计算用量 |
| 结论 | 分别给两个任务的原版/Omni/差值；按 folding 目标或 design 任务做配对 bootstrap；两项独立提高才支持 PoC，区间跨零则不确定 |

来源限制：Boltz-2 官方 evaluation.md 仍把更新后的 Figure 15 测试文件标为
“Coming soon”。本次 folding 用作者实际公开的单体验证名单，**不称作 Figure 15
测试集复现**。序列单独输入，不带 MSA。全原子 lDDT 使用固定原子名对应，不做侧链
对称置换。Design 指标衡量 ProteinMPNN 重设计后的骨架可设计性，不能单独证明
新增序列头变强。样本小，结果范围限于这个筛查子集。

Quest 每次提交时分别查询 p53127、p53252 中 jck3114 的当前 FairShare；用
sbatch --test-only 核实请求资源的 eligibility 后选择较高者，不沿用旧快照。

前置数值检查补充：跨 H100/A100 重用缓存时，重新计算当前设备上的冻结教师
conditioning bridge，使零反馈学生与教师逐元素一致（实测最大误差 0）。
TF32 下刚体坐标旋转实测造成最大 0.010757 Å 的距离变化，因此坐标旋转和采样
坐标系累计/逆变换使用 FP64，随后转换回模型输入精度；神经网络仍使用既定精度。
这些是参数更新前的实现检查，不计作训练成功或效果改善。

原始失败训练的定量检查：固定 validation design 坐标 MSE 从 step 250 的
0.6156 降到 step 1000 的 0.2623，异常相邻 Cα 间距从 3.0 增至 65.5；
说明旧 loss 改善不能验证采样正确。已确认的上述实现差异需要修复；修复是否足够
只能由新训练和完整配对评测回答，不能预先保证结果正向。

评分实现核验：重折叠运行期间独立检查发现，Scaffold-Lab 的 rmsd 列实际为刚体
对齐后 Cα 距离的算术平均。最终汇总前从已保存 PDB 统一重算约定的真正 RMSD，
同时重算未取整的 motif RMSD；不改变阈值、候选、模型或置信度指标。原始评分表
完整保留，修正记录写入 design_rescoring.json 并上传 W&B。
