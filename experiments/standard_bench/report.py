"""Paired public-subset report: independent tasks, fixed checkpoint, no training."""
import csv
import json
from pathlib import Path
import numpy as np
from tracking import Tracker,file_hash

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def paired(a,b,higher_is_better=True):
    a=np.asarray(a,float);b=np.asarray(b,float);d=b-a
    rng=np.random.default_rng(20260927)
    means=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
    return dict(native=float(a.mean()),omni=float(b.mean()),delta=float(d.mean()),
        paired_bootstrap_95_ci=np.quantile(means,[.025,.975]).tolist(),
        units=len(d),higher_is_better=higher_is_better,
        wins=int(((d>0) if higher_is_better else (d<0)).sum()),
        ties=int((d==0).sum()),losses=int(((d<0) if higher_is_better else (d>0)).sum()))

def save_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    audit=json.loads((ROOT/'input_audit.json').read_text())
    assert audit['passed'] and audit['manifest_sha256']==file_hash(ROOT/'manifest.json')
    folds=[]
    for target in manifest['folding']:
        record=dict(target=target['id'],length=target['length'],resolved_ca=len(target['resolved_ca']))
        for method in ('native','omni'):
            r=json.loads((OUT/f'{target["id"]}-{method}.json').read_text())
            for metric in ('ca_lddt','tm_score'):record[f'{method}_{metric}']=r[metric]
        for metric in ('ca_lddt','tm_score'):record['delta_'+metric]=record['omni_'+metric]-record['native_'+metric]
        folds.append(record)
    designs=json.loads((OUT/'design_scores.json').read_text())
    cases=[]
    for target in manifest['motif_cases']:
        record=dict(case=target['case'],group=target['group'],length=target['length'],candidates_per_model=2)
        for method in ('native','omni'):
            matches=[r for r in designs if r['case']==target['case'] and r['method']==method]
            assert len(matches)==1 and matches[0]['total']==2
            record[method+'_success']=matches[0]['success']
            record[method+'_rate']=matches[0]['success']/2
            record[method+'_best_sc_rmsd']=float(np.mean([s['best_sc_rmsd'] for s in matches[0]['scaffolds']]))
        record['delta_rate']=record['omni_rate']-record['native_rate'];cases.append(record)
    summary={metric:paired([r['native_'+metric] for r in folds],[r['omni_'+metric] for r in folds])
        for metric in ('ca_lddt','tm_score')}
    summary['design_success']=paired([r['native_rate'] for r in cases],[r['omni_rate'] for r in cases])
    summary['design_best_sc_rmsd']=paired([r['native_best_sc_rmsd'] for r in cases],[r['omni_best_sc_rmsd'] for r in cases],higher_is_better=False)
    summary['design_best_sc_rmsd']['interpretation']='Supplementary descriptive readout from official outputs; lower is better. Average per-scaffold best over 8 sequences, with uncertainty resampled by task. Does not replace the primary joint success criterion.'
    summary['design_counts']={m:sum(r[m+'_success'] for r in cases) for m in ('native','omni')}
    summary['design_counts']['denominator_per_model']=12
    summary['scope']='Small fixed retrospective public subset; no full leaderboard claim; base pretraining exposure not excluded.'
    summary['design_scope']=manifest['design_scope']
    summary['positive_rule']='Both independent task point estimates improve; robust claim requires paired uncertainty excluding zero for both. A small subset can be inconclusive.'
    summary['shared_checkpoint']=manifest['checkpoint']
    summary['manifest_sha256']=file_hash(ROOT/'manifest.json')
    compute_path=OUT/'compute_usage.json'
    if compute_path.exists():summary['compute']=json.loads(compute_path.read_text())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    save_csv('folding_scores.csv',folds);save_csv('design_task_scores.csv',cases)
    lines=['# 小规模标准 benchmark：原版 BoltzGen vs 当前 Omni','',
        '固定使用同一个 final.pt（1000 步），没有新增训练；每题采用相同输入、生成种子及采样步数。','',
        '| 能力 / 标准指标 | 原版 | Omni | 差值（Omni − 原版） | 配对 bootstrap 95% 区间 |',
        '|---|---:|---:|---:|---:|']
    for key,label in [('ca_lddt','CASP16，10 题，Cα lDDT ↑'),('tm_score','CASP16，10 题，TM-score ↑'),('design_success','MotifBench，6 题 × 2 候选，成功率 ↑'),('design_best_sc_rmsd','补充：每个候选最佳 scRMSD 均值，Å ↓')]:
        s=summary[key];lo,hi=s['paired_bootstrap_95_ci']
        lines.append(f'| {label} | {s["native"]:.4f} | {s["omni"]:.4f} | {s["delta"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    counts=summary['design_counts']
    lines+=['',f'Design 成功数：原版 {counts["native"]}/12，Omni {counts["omni"]}/12。',
        '', 'Design 使用官方 MotifBench ProteinMPNN（每个 scaffold 8 条序列）+ ESMFold，成功要求同一条回折结果同时满足 scRMSD < 2 Å、motif RMSD < 1 Å。没有另加身份一致性过滤。',
        '', '补充的连续读数直接来自官方 scRMSD 输出：每个候选取 8 条评估序列中最小的 scRMSD，再平均。它描述与设计骨架偏离多大，不替代同时满足两个 RMSD 阈值的成功率主指标。',
        '', 'Folding 的 lDDT 仅计算实验结构中已解析的 Cα，采用 Biotite 标准实现；TM-score 采用 TM-align。两者均为 0–1 标度。这不是完整 CASP 官方排名。',
        '', '抽样在预测前锁定：CASP16 按长度取 3/4/3 题并去掉适配训练/验证的序列同源命中；MotifBench 三个片段数分组各取短、长两个不同 PDB 的任务。完整题目和逐题结果见同目录 CSV 与 manifest.json。',
        '', '解读：只有 folding 和 design 各自提升，才能支持“双任务都更强”。区间跨零则该小样本不足以下稳定结论；两边全零时 bootstrap 零区间不表示能力相同，而表示该测试未分辨出成功率差异。设计区间按 6 个任务重采样，不把 8 条配序列当独立题目。',
        '', '范围：MotifBench 检验 scaffold 设计能力，序列经 ProteinMPNN 重设计，因此不能单凭它证明 Omni 自身序列头更好。原版预训练是否见过这些公开目标未排除，结果是配对回溯比较，不是盲测泛化证明。',
        '', '来源：[CASP16 公共目标](https://predictioncenter.org/download_area/CASP16/)；[MotifBench 官方代码与定义](https://github.com/blt2114/MotifBench)。']
    if 'compute' in summary:
        lines+=['',f'本次标准测试累计分配 GPU {summary["compute"]["total_GPU_hours"]:.3f} 小时（含中断后续跑），没有新增训练。时间来自 Slurm 实际作业记录，排队时间不计入。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tracker=Tracker(OUT/'summary-tracking',dict(manifest_sha256=file_hash(ROOT/'manifest.json'),
        source_sha256=file_hash(__file__),training_performed=False,shared_checkpoint=manifest['checkpoint']),
        'standard-benchmark-paired-results',job_type='evaluation')
    try:
        tracker.run.config.update({'source_sha256':file_hash(__file__)},allow_val_change=True)
        tracker.log({f'{key}/{k}':v for key in ('ca_lddt','tm_score','design_success','design_best_sc_rmsd') for k,v in summary[key].items() if isinstance(v,(int,float))},0)
        if 'compute' in summary:tracker.log({'compute/total_GPU_hours':summary['compute']['total_GPU_hours']},0)
        import wandb
        artifact=wandb.Artifact('standard-benchmark-paired-report',type='evaluation')
        for p in (OUT/'summary.json',OUT/'folding_scores.csv',OUT/'design_task_scores.csv',ROOT/'RESULTS.md',ROOT/'manifest.json',ROOT/'input_audit.json',ROOT/'compatibility_changes.json',ROOT/'evaluator_packages.txt'):artifact.add_file(str(p))
        for p in ROOT.glob('*.py'):artifact.add_file(str(p),name='evaluation_sources/'+p.name)
        for p in (compute_path,OUT/'verification.json'):
            if p.exists():artifact.add_file(str(p))
        tracker.run.log_artifact(artifact)
    finally:tracker.finish()
    print('PAIRED_STANDARD_BENCHMARK_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
