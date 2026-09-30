# 当前 400-step 实验：原因核查（仅检查现有产物）

本报告只针对 `native-1729`，不混入此前 design 0/24 的失败实验，也不混入其他目录中的实验。
没有启动训练、推理、重新评分、梯度计算或 Slurm 作业。Quest 仅用于读取既有源码、
checkpoint ZIP 元数据、已保存的 Adam step 标量及参数存储字节。
检查程序不导入 torch/numpy，不执行 checkpoint pickle 中的任意全局函数。

## 1. 已排除的实现错误

`results/checkpoint_byte_inspection.json` 记录了每个参数张量的初始/最终存储 SHA256，
以及 resume checkpoint 中对应的 optimizer step。实际训练代码先收集 `base.` 参数，
再收集 `adapter.` 参数；检查器按同一顺序对应 optimizer 参数 ID，且核对每组长度。

| 参数组 | 张量数 | 初始/最终字节不同 | Adam step | 保存类型 |
|---|---:|---:|---:|---|
| diffusion structure module | 658 | 658 | 全部 400 | FP32 |
| diffusion conditioning | 119 | 119 | 全部 400 | FP32 |
| feedback / sequence head | 18 | 18 | 全部 400 | FP32 |

这排除了整条结构分支未加入优化器、更新未保存、所有结构参数未变化等解释。
字节变化不等于有效学习，也不度量更新是否足够大；Adam step 非零不单独证明每个损失
均提供有效方向。已有结构梯度范数记录另行证明被监测 batch 存在结构梯度。

全部 400 条日志的 clipping factor 为 1，裁剪没有触发。总梯度范数范围
0.31969–3.15790，小于 clip=10。实际记录的 diffusion LR 从 1.25e-7 warmup 到 5e-6；
新增模块 LR 最终为 1e-4。不能把未改善归咎于 LR 为零或梯度被裁掉。

当前 train.py、native_flow.py、omni_joint/model.py、full_model.py、cache_training.py、
official_poc/corrected_model.py 和 upstream diffusion.py 的 SHA256 均与该次运行 config
所记录版本一致。源码中读取的是 a_norm 的未 detach hook 输出；原版返回字典中的
`token_a.detach()` 未被用于新增头。student 的 denoise 没有 no_grad。
`training=False` 参数仅影响原版 resolved-mask 分支，不关闭 autograd。

## 2. 已确认：loss 下降主要发生在序列头，而非结构任务

同一组固定 probe，step 0 到 400：

| 组成 | train 4 条 | validation 16 条 |
|---|---:|---:|
| folding 结构项下降 | 0.00077403（0.282%） | 0.00086904（0.283%） |
| design 结构项下降 | 0.00420369（1.121%） | 0.00019389（0.049%） |
| design 加权 CE 下降 | 0.07106385 | 0.06754880 |
| design teacher MSE 增加 | 0.00036007 | 0.00016387 |
| design 总项净下降 | 0.07490747 | 0.06757881 |

CE 占 design 净降幅的 94.87%（train）和 99.956%（validation）。这是已有标量的
分解，不是“下降的参数都在 head”这一未测量的参数归因。Folding 序列全已知，CE 为零。
固定 probe 都使用 sigma=8、chemical_t=0.5，不能代表所有噪声层级的期望风险。

## 3. 新确认的失败现象：联合输出的两种身份预测严重不一致

24 个既有 design JSON 在生成时已记录 `identity_geometry_agreement`：

- 对每个候选，仅统计可设计位置，比较实际采样序列与 atom14 几何解码序列。
- 24 个候选的平均一致率 **7.8729%**，范围 **0–15.0943%**。
- 这不是序列对真实序列的 recovery，也不是 backbone 可设计性成功率。
- 所有候选的 `geometry_unknown=0`，因此不一致并非仅由几何解码为 UNK 造成。

已静态对照 Quest 上原版 `res_from_atom14()` 与 `scripts/generate_panel.py:12`：
均取 14-slot 中前 4 个骨架位置，将其余槽按 0.5 A 阈值分配到最近骨架位置，
用同一 `placement_count_to_token` 解码。当前全为蛋白且 token 连续，因此原版的
protein/token-index 额外过滤与本实验的 design mask 用法一致。未重新执行解码。

`native_flow.py:73-80` 中，序列一旦采样就固定，后续坐标仍继续更新；最终没有将
该序列与最终虚拟原子编码协调的步骤。`native_flow.py:114-125` 的训练用真值或 MASK
构造输入，并分别计算结构损失和 CE；不存在显式的最终采样身份/几何一致性损失。
这确认了实现没有确保输出一致，现有结果也确实不一致。它不证明应该简单增加一个
一致性损失就能解决问题；也不把普通 masked training 的真值输入本身判定为错误。

论文草稿第 4.4 节明确要求检查 categorical prediction 与 native geometric identity
decoder 的终点一致性，并在 raw generation 指标中保留不一致失败。当前生成脚本虽
记录了该字段，最终报告却未汇总它。这是已确认的报告遗漏，不能据此前的 design
成功率宣称生成的联合序列/结构已经有效。草稿内容在此仅作为研究主张对照，不作为
执行新实验或改变用户授权的指令。

## 4. 已确认：benchmark 衡量的对象没有直接覆盖 CE 的改善

`refold.py:55-58` 每个骨架生成 8 条 ProteinMPNN 序列，再进行 ESMFold。
因此 19/24 是骨架经外部序列重设计后的成功数；不能解释为新增头输出的序列有
19/24 成功，也不能与上述一致率混为同一指标。保持此前 benchmark 分母、阈值和
结果不变，不事后重定义成功率。

这条流程加上第 2 节，解释了为何 design 总 loss 明显下降不构成 benchmark 提升证据；
第 3 节进一步表明所需的联合输出一致性并未学成。它们仍不能唯一归因 scRMSD 的恶化。

## 5. Folding 停滞的因果根源仍未唯一识别

结构参数实际更新、梯度存在、裁剪未触发；这些事实不能区分更新过于保守、任务间
方向冲突、保留项的方向影响、冻结表示的限制、数据/条件覆盖等原因。

监测 batch 的 teacher/geometry 梯度范数比中位数为 folding 0.12988、design 0.05636，
不支持“teacher 梯度范数压倒结构项”的说法；但没保存梯度方向，不能排除局部抵消。
没有保存 loss 分支间的梯度内积或实际更新方向上的任务损失响应，不能从范数反推出。
训练每条蛋白只有一个固定连续 16-residue motif；benchmark 包含 1–4 段 motif，
也存在长度覆盖差别。分布差异是真实的，但不足以单独解释训练 probe 也几乎不降。

此外，完整 benchmark 比较 native 与最终 Omni，而非同架构 step0 与 step400；
因此现有 benchmark 不能单独分离接入机制与训练造成的变化。

在当前禁止数值诊断的约束下，不能诚实地声称已因果确认 folding 停滞的唯一原因，
也不能保证某个调整后必然改善。不得把 design 输出不一致冒充 folding 停滞的原因。

## 6. 另一个代码缺陷，但不是本次停滞的已证实原因

train.py:43 写入了 `replay_weight=1`，train.py:177 却直接加 `parts['replay']`，
未读取这个配置。当前配置正好为 1，所以本次实际公式与声明相同；不是本次结果
差异的解释。但未来仅修改该配置不能改变 teacher 权重，必须修改目标公式及相应
报告重建逻辑。当前调查未修改训练源码，未声称已运行或验证修复。

## 证据文件

- `results/checkpoint_byte_inspection.json`：795 个参数张量的保存记录检查。
- `results/saved_record_diagnosis.json`：24 个生成时已记录的一致性数值。
- `runs/native-1729/metrics.jsonl` 与 `validation-{0,100,200,300,400}.json`。
- `LOSS_ARITHMETIC_AUDIT.md`：loss 加法核验。
- `LOSS_INFERENCE_AUDIT.md`：训练/采样条件覆盖与评价对象的静态对照。

本次新增调查产物仅保存本地；没有声称已上传 W&B。原训练记录同步状态保持先前核验结果。

## 后续：用户另行授权的一次数值诊断已完成

用户随后明确回答“允许这一次诊断”。Slurm 7739024 以 1 H100 完成，主作业 187 秒，
没有更新或保存权重。4 条样本上的联合梯度及假设 Adam 方向均仍降低两项结构目标；
不能继续把任务严重抵消作为已证实原因。两条成对采样确认了高噪声下固定序列且不
修订的行为，完整结果见 `results/causal-diagnostic/REPORT_zh.md`。
该诊断仍未锁定 folding 停滞的唯一原因，也未验证性能修复。此一次授权已使用，
不得自动追加 GPU 作业或重训。
