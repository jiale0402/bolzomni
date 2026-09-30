# 已完成的静态核查；不是性能验证

## 原始联合输出与MPNN指标

草稿4.4节写明：分类身份与原生几何身份解码必须在终点检查一致性，非法或不一致的
输出计失败，不能先修复再计原始指标。旧复制的评测只报告一致率并使用MPNN重新设计，
不足以验证原始联合输出。现在保留原benchmark，同时新增独立的raw_design结果：
不一致等失败保留在24个候选分母中；有效者使用未经替换的模型序列进行ESMFold。
同样的四项质量阈值，单序列与best-of-8分开；该补充指标不能替代既定benchmark。

原始ESMFold调用核对了Quest现装Scaffold-Lab/motif_refolding.py第687–698行，
使用同一个infer/output_to_pdb方法；pAE和pLDDT读取与该文件第567–569行相同的字段。
RMSD仍调用已核实的root-mean-square评分，未使用vendor的mean-distance冒充RMSD。

## 数据表示核查

cache_training.py把原始atom_resolved_mask作为训练监督mask。现装featurizer.py
第1088–1106行给atom14虚拟O/N位置同时复制其母原子的is_present，第1402行再
构造resolved_mask。因此源码不支持“虚拟身份槽全部被resolved mask排除”这一猜测。
没有重新运行featurizer或模型，不能据此声称每个现存cache均通过新的数值检查。

## 续训与生成一致性

从保存的400步Adam/RNG/数据顺序继续，新增源代码与原历史model/loss源码分目录。
新增静态哈希校验限制历史model/loss漂移；实际有限步loss比较保留8个microbatch
及各自RNG，更新后按同批同噪声重算，随后恢复训练RNG。

成对生成新增provenance记录，禁止把不同checkpoint、manifest、源文件或Slurm作业
输出混合后声称是同一GPU比较。报告校验全部候选ID、种子与checkpoint hash。
报告异常时W&B不再被标记成功。尚未进行模型兼容性或性能验证。

## 未解决项

Folding停滞的唯一原因仍未查明。CDF时间表是符合Algorithm1规则的候选选择，
不是论文指定公式，也未验证有效。延长训练没有保证改善。新的模型运行需要回答
已发送的累计2 GPU-hour预算问题；一次诊断授权已用完。
