# 一次已授权的原因诊断

- 用户明确授权：“允许这一次诊断”。
- Slurm 7739024，p53252，gengpu，1 H100，硬时限 15 分钟。
- 提交时间：2026-09-28 22:19:08 UTC。
- 提交时 jck3114 FairShare：p53252 0.442961，p53127 0.427890；两者资源资格均通过。
- 提交证据：`submissions/7739024.json`。
- 最终状态：COMPLETED / 0:0；主作业 187 秒（0.0519444 GPU-hour），extern 子步骤 190 秒，未重复相加。
- 范围：4 条原 train probe 的分项梯度诊断；5YUI_0、6E6R-short_0 两个设计案例的成对反馈轨迹。
- 禁止 optimizer.step、保存新模型权重、重训、额外 benchmark、自动续交或追加 GPU 作业。
- 诊断输出目录：`results/causal-diagnostic`；源文件、输入和 checkpoint hash 将随 W&B/JSONL 保存。
- 已完成 4 条梯度检查、4 条完整采样轨迹。权重未更新、未保存新 checkpoint。
- W&B run 69bd03743201 已核实 finished；原始诊断及最终报告两个 artifact 均 COMMITTED，完整 Slurm 用量 187 秒已同步。
- final.pt SHA256 再次核实未变；完整生成记录、NPZ、报告及同步核验均已下载本地。
- 报告：`results/causal-diagnostic/REPORT_zh.md`。未确定 folding 停滞的唯一原因，未声称验证性能修复。

如果诊断失败或预算耗尽，保留部分结果并报告，不静默重新提交。
