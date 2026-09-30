# 已授权的一次最小原因诊断

授权：用户于本线程明确回答“允许这一次诊断”。仅解除本方案范围内的一次诊断
compute 限制；不授权重训、保存新权重、增加任务或在超时后自动续交。
执行状态以 Slurm、诊断记录和 W&B 为准，本文不预先声称诊断已完成。

对象：现有 native-1729 初始、最终及 optimizer checkpoint；不进行新训练，
不选择新 checkpoint，不修改已完成实验源码，不运行 MPNN/ESMFold benchmark。

## A. Folding 的结构目标有没有被联合更新抵消

- 固定已有 4 条 train probe，逐条复用原先的种子、sigma=8、chemical_t=0.5。
- 在最终 checkpoint 分别计算 folding 结构、design 结构、加权 CE、teacher
  四类梯度；按实际任务权重组合，检查对 folding / design 结构目标的方向导数。
- 分开报告 diffusion、conditioning、feedback/head；同时记录梯度范数、内积、
  cosine，以及 Adam 预条件方向。方向导数仅为局部证据，不能冒充全程归因。
- 不执行 optimizer.step，不保存新权重；不只根据梯度范数判断任务冲突。
- 结果必须能回答：在这些具体状态和样本上，哪项贡献让结构目标的下降方向减弱
  或反转。若没有冲突，应明确排除这批样本上的冲突解释，不能认定改 lambda 有效。

## B. Design 的自生成序列反馈是否损害生成

- 两个预先固定旧案例：5YUI_0（原面板主要退化任务之一）、6E6R-short_0。
- 同一最终 checkpoint、同一初始噪声和原生 500-step 坐标 sampler，成对检查
  原 joint 轨迹与保持未知序列为 MASK 的诊断轨迹。后者仅用于定位反馈作用，
  不作为新模型或论文结果，也不替换正式输出。
- 保存当前 logits、已揭示序列与几何身份的一致性随时间变化、结构有限性、
  相邻 CA 距离和 motif 条件误差。不重跑外部 refolding。
- 这个比较能定位条件反馈对具体输出的影响，不能单独证明通用 designability
  改善；没有 MPNN/ESMFold 新评测，就不能报告新的成功率。

## 预算和留痕

- 仅一个 Quest GPU 作业，申请 1 GPU，Slurm 硬时限 15 分钟（最多 0.25 GPU-hour）。
- 提交前现场比较 p53127 / p53252 的 jck3114 用户 FairShare，并确认资源资格。
- 原始诊断记录、源码/输入/checkpoint hash 保存 JSONL 与 W&B。
- 时间耗尽保留已完成诊断，不自动续交、不训练、不改正式模型；报告未完成项。
- 不保证这一次诊断能锁定唯一原因；先报告支持/排除的机制，再决定修复内容。
