# 目标：loss 与 folding / design benchmark 同时改善

状态：目标未完成。前一轮诊断已完成，是有效进展；没有把诊断完成视为科研目标完成。
历史 400-step 实验与最终 checkpoint 保留。旧目录中的 COMPLETED 仅表示旧运行结束。

## 本轮改动及论文对应

| 元素 | 具体安排 |
|---|---|
| 核心机制 | 一个 frozen trunk、共享 diffusion、反馈模块和序列读出；不切换任务模型 |
| 训练参数 | 按用户既定要求全参数训练 diffusion/conditioning/反馈/序列头，冻结 trunk，无 LoRA。明确这与草稿默认 LoRA 参数范围不同，但保留其联合补全机制 |
| 训练目标 | 原生结构 MSE + smooth-lDDT + masked identity CE + teacher MSE；两任务共用公式，只改变观察字段；保留已校准 CE 权重，不根据测试集调 lambda |
| 权重配置修正 | teacher 权重明确进入实际公式，默认保持 1；不能只改配置而实际仍固定为 1 |
| 采样修正 | 原生坐标 solver 不动；以训练噪声分布 CDF 构造 chemical t，仍严格用论文 Algorithm 1 的 1-t_next/t_current 揭示概率 |
| 训练噪声 | sigma 与 chemical t 继续独立采样，保留 fully masked / fully visible 端点；采样时间表不改写训练的独立噪声假设 |
| 新时间表的地位 | 是诊断驱动的实现选择，非论文指定公式；尚无性能改善证据 |
| Folding 未决项 | 梯度方向下降不等于有限步实际下降；必须在同一训练批次和随机数下记录真实更新前后结构 loss，区分优化步效应与换 batch 噪声 |

## 拟申请的后续计算范围

上次授权仅覆盖已完成的一次 15 分钟诊断。新的 goal 不被用来自动解释为无限 GPU 预算。
拟申请 Quest 累计最多 2 GPU-hour；不获授权不提交。

1. 一条 Omni 续训路径，从已保存的 step400 权重及 Adam 状态继续，绝不静默从零开始。
   先实现固定随机数下的实际 update 前后观测，再在同一续训路径继续。没有原架构
   continued-training control，没有多组训练消融，没有候选 checkpoint 按测试分数挑选。
2. 数据保持已验证的 128 条官方训练链及 16 条验证链。从 update400 连续到 update2000，
   追加 1600 次更新，固定最后一个 checkpoint。原 Adam 状态、数据顺序、随机数状态、
   diffusion LR=5e-6、新模块 LR=1e-4、CE 权重=0.1896582197810322、teacher 权重=1
   全部继续，不重新 warmup 或校准。延长训练是增加结构目标的训练暴露，不是已证实的
   folding 修复；若结构 loss 或 benchmark 不改善，必须如实报告。
3. 相同 shared checkpoint 评估两任务。保持原 8 个作者公开 folding 验证目标（各 5 次）
   和 6 个 RFdiffusion 任务（各 4 骨架、各 8 条 MPNN/ESMFold 序列），保留全部失败。
   这是已被查看过的探索性面板，不宣称新的确认性测试集或 Figure15 测试集。
4. 保留骨架经 MPNN 重设计的主benchmark，同时增加原始联合输出评测，各模型24个。
   依草稿4.4节，序列与几何身份不一致、未知身份、非有限几何或观察身份被破坏的候选
   直接计失败，绝不先更换序列。有效候选直接将原始序列送入同一ESMFold，用相同四项
   数值阈值评分，最多额外48条重折叠。该单序列结果与MPNN best-of-8分开给出。
   原始输出仍无改善时，即使MPNN骨架分数提高，也不能称为论文联合输出能力得到验证。
5. 总预算包含失败尝试、初始化、生成与 refolding，按 Slurm 分配时间计；不自动超额
   或无界搜参。每次提交现场比较两个 allocation 的用户 FairShare 和资源资格。
   拟分配训练+成对生成硬上限 90 分钟，MPNN及原始序列refolding+汇总硬上限30分钟，各1张GPU；
   禁用 requeue。超时保留已完成参数更新，不能用较早权重冒充固定 update2000。
   原版和 Omni 在同一生成作业内用同一 GPU 重新生成，消除先前 A100/H100 差异。
6. 第401步及其后每100步，在真实 optimizer.step 前后，用同一8个microbatch和随机数
   记录 geometry / CE / teacher / total 的变化。诊断不额外更新权重，恢复训练随机数。
   每200步记录 train4 / validation16 的固定探针，保持 sigma=8、chemical_t=0.5
   以便与历史曲线直接比较；同时记录全部训练microbatch的sigma和t，不把单个探针
   当作整个训练分布的结论。

## 不变的成功标准

- train / validation 的 folding 结构项、design 结构项、序列项、teacher 项、加权总项
  分开报告；固定样本和噪声上做前后比较，不仅宣称总 loss 下降。
- Folding：保留 best-of-5 CA-lDDT、TM-score，报告逐题差值与目标级配对区间。
- Design：保留官方四条件成功率及 best-of-8 scRMSD，报告任务级配对区间。
- 一个 checkpoint 对未训练原版。两任务需要各自有改善证据，不能拿序列 CE
  或局部梯度方向替代 benchmark。区间不支持提高时不能声称已经验证提高。
- 固定原有阈值、失败分母和报告口径；不得靠更换主指标、移除难题或重新定义成功过关。
- W&B + 本地 JSONL 记录源码、数据、权重、全部 loss、选点规则及完整 GPU 用量。
- 实验完成不等于此目标完成；目前没有满足联合改善的证据。
