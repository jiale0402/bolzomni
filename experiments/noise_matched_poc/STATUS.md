# 等待新增计算授权，尚未提交或运行

科学目标仍是一个共享模型的 loss 下降、folding 和 design benchmark 各自改善。
尚无新结果，不能声称修复有效或目标完成。

本目录是新的候选实现，不覆盖 native_condition_poc 的历史代码、日志和权重。

- chemical_schedule.py / noise_matched_flow.py：按训练坐标噪声的 CDF 选取化学
  时间表，保留论文 1-t_next/t_current 规则；保留原生坐标 sampler。
- train_continue.py：从保存的 update400 权重、Adam、数据顺序及 RNG 接着训练；
  追加1600次；每隔100步记录一次真实更新前后同批同噪声分项 loss。
  时间告警在更新边界保存；无质量恶化停训条件，无静默重新初始化。
- generate_pair.py / refold.py / report.py：同一 GPU 上重新生成原版和最终 Omni，
  沿用8×5 folding及6×4×8 design题目、种子、阈值、目标/任务级配对区间。
  MPNN成绩只作为骨架可设计性指标；补入raw_joint.py及原始序列独立refolding代码。
  不一致的联合输出直接计失败，有效者不换序列直接用同一ESMFold评估；最多额外48条，
  保持30分钟refold阶段硬上限。不能将MPNN的成功称为新序列头成功。
- benchmark_manifest.json：仅更新checkpoint及续训说明；题目、随机种子、阈值未改。
- train_generate.sbatch / refold_report.sbatch：拟定90+30分钟、各1张H100，禁止requeue。
  两次提交均需现场比较jck3114两个allocation的FairShare及资源资格。

已做的检查仅包括源码审查、Python AST语法检查、先前4项纯标量时间表测试。
没有加载模型，没有执行新增训练/推理/评估，没有上传或提交本目录。
本次续接仅通过SSH读取已安装featurizer与ESMFold调用源码，未启动Slurm或模型。
尚未验证模型运行兼容性、耗时或性能恢复。

授权：一次15分钟诊断7739024已完成，用量187秒；该授权已用完。
额外累计2 GPU-hour的async预算问题仍待用户回答。未获答复不得提交。
不得因goal自动继续或超时就解释为允许计算。

阻塞审计：前两次goal turn分别完成候选训练/评测代码、补齐草稿4.4节原始联合输出
评分，均为progress。本次是连续第三次等待同一额外计算授权，仍未收到答复。
已现场只读查询Quest：7739024仍为COMPLETED/0:0/187秒；squeue没有本诊断或新
续训/重折叠作业，远端noise_matched_poc目录尚不存在。没有正在运行的本目标进程可等待。
安全的准备工作已完成；下一项能验证loss及benchmark改善的工作需要模型计算。
将目标设为blocked，等待用户回答现有2 GPU-hour预算问题；目标未完成，不自动提交。

剩余执行检查：获授权后先上传并核实完整源码/manifest；新提交必须查询实时FairShare；
完成后读取主作业sacct用量（包含失败及导入开销），核实W&B finished及artifact COMMITTED，
下载全部逐题分数、原始记录及报告。任一阶段未完成必须报告缺失分母，不能只汇总成功文件。
