"""Require complete paired denominators; publish both tasks and training diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
from tmtools import tm_align
from tracking import Tracker,file_hash

ROOT=Path(__file__).resolve().parent


def paired(native,omni):
    a=np.asarray(native,dtype=float);b=np.asarray(omni,dtype=float);delta=b-a
    rng=np.random.default_rng(1729)
    boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
    return dict(native=float(a.mean()),omni=float(b.mean()),delta=float(delta.mean()),
        ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),units=len(delta))


def main():
    output=ROOT/'results';manifest=json.loads((ROOT/'benchmark_manifest.json').read_text())
    records=[json.loads(p.read_text()) for p in output.glob('*-native.json')]+[json.loads(p.read_text()) for p in output.glob('*-omni.json')]
    folds=[r for r in records if r['mode']=='folding'];designs=[r for r in records if r['mode']=='design']
    assert len(folds)==80 and len(designs)==48,(len(folds),len(designs))
    design_scores=json.loads((output/'design_scores.json').read_text());assert len(design_scores)==12
    fold_rows=[]
    for row in manifest['folding']:
        for method in ('native','omni'):
            chosen=[r for r in folds if r['target']==row['id'] and r['method']==method]
            assert len(chosen)==5 and len({r['seed'] for r in chosen})==5
            fold_rows.append(dict(target=row['id'],method=method,
                **{'best5_'+metric:max(r[metric] for r in chosen) for metric in ('ca_lddt','all_atom_lddt','tm_score')},
                **{'mean5_'+metric:float(np.mean([r[metric] for r in chosen])) for metric in ('ca_lddt','all_atom_lddt','tm_score')}))
    summary=dict(folding={},design={},limitations=[manifest['folding_limitation'],
        '8 folding structures and 6 design tasks are a small screening panel, not a full leaderboard or general superiority proof.',
        'This is a previously inspected exploratory regression panel; unchanged baseline results are reused.',
        'Design success evaluates scaffold designability after ProteinMPNN; it does not establish sequence-head quality.',
        'All-atom lDDT uses native lddt_dist and fixed atom-name correspondence without side-chain symmetry optimization.'])
    for metric in ('best5_ca_lddt','best5_all_atom_lddt','best5_tm_score','mean5_ca_lddt','mean5_all_atom_lddt','mean5_tm_score'):
        summary['folding'][metric]=paired([r[metric] for r in fold_rows if r['method']=='native'],[r[metric] for r in fold_rows if r['method']=='omni'])
    drows=[]
    for case in manifest['design_cases']:
        task=case['case']
        for method in ('native','omni'):
            row=next(r for r in design_scores if r['case']==task and r['method']==method)
            assert row['total']==4 and len(row['scaffolds'])==4 and all(s['sequence_count']==8 for s in row['scaffolds'])
            success=[r for r in row['scaffolds'] if r['success']]
            parent=list(range(len(success)))
            def root(i):
                while parent[i]!=i:i=parent[i]
                return i
            for i in range(len(success)):
                x=np.load(output/f'{success[i]["id"]}-{method}.npz')['backbone'][:,1].astype(float)
                for j in range(i):
                    y=np.load(output/f'{success[j]["id"]}-{method}.npz')['backbone'][:,1].astype(float)
                    score=tm_align(x,y,'A'*len(x),'A'*len(y))
                    if min(score.tm_norm_chain1,score.tm_norm_chain2)>=.6:parent[root(i)]=root(j)
            drows.append(dict(task=task,method=method,successes=row['success'],total=4,success_rate=row['success']/4,
                unique_successes=len({root(i) for i in range(len(success))}),best_sc_rmsd_mean=float(np.mean([s['best_sc_rmsd'] for s in row['scaffolds']]))))
    summary['design']['success_rate']=paired([r['success_rate'] for r in drows if r['method']=='native'],[r['success_rate'] for r in drows if r['method']=='omni'])
    summary['design']['best_sc_rmsd_mean']=paired([r['best_sc_rmsd_mean'] for r in drows if r['method']=='native'],[r['best_sc_rmsd_mean'] for r in drows if r['method']=='omni'])
    rmsd=summary['design']['best_sc_rmsd_mean'];rmsd['wins'],rmsd['losses']=rmsd['losses'],rmsd['wins']
    summary['design']['totals']={m:dict(successes=sum(r['successes'] for r in drows if r['method']==m),total=24,unique_successes=sum(r['unique_successes'] for r in drows if r['method']==m)) for m in ('native','omni')}
    summary['counts']=dict(folding_targets=8,folding_predictions=80,design_tasks=6,scaffolds=48,refolded_sequences=384)
    f=summary['folding']['best5_ca_lddt'];d=summary['design']['success_rate']
    summary['both_point_estimates_improve']=f['delta']>0 and d['delta']>0
    summary['both_ci_exclude_zero_above']=f['ci95'][0]>0 and d['ci95'][0]>0
    summary['checkpoint']=manifest['checkpoint'];summary['checkpoint_sha256']=file_hash(ROOT/'runs/native-1729/final.pt')
    training=[json.loads(l) for l in (ROOT/'runs/native-1729/metrics.jsonl').read_text().splitlines()]
    summary['training_windows']={}
    for a,b in [(1,40),(41,100),(101,200),(201,300),(301,400)]:
        chosen=[r for r in training if a<=r['optimizer_step']<=b and 'train/loss' in r]
        assert len(chosen)==b-a+1
        summary['training_windows'][f'{a}-{b}']={k:float(np.mean([r[k] for r in chosen])) for k in ('train/loss','train/folding/loss','train/design/loss','train/gradient_norm_before_clip','train/gradient_clip_factor')}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    for name,rows in [('folding_scores.csv',fold_rows),('design_scores.csv',drows)]:
        with (output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=f'''# 官方数据子集 PoC 实测结果

这是已查看过的探索性修复回归子集，原版结果复用已核验结果。

同一个 Omni 最终 checkpoint，第 400 次更新；对照为未训练的原版 BoltzGen。
训练数据：官方发布训练归档中的 128 条完整蛋白链，16 条独立验证链。

| 指标 | 原版 | Omni | Omni − 原版 | 配对 95% 区间 |
|---|---:|---:|---:|---|
| Folding Cα-lDDT，8 条、每条 best-of-5 | {f['native']:.4f} | {f['omni']:.4f} | {f['delta']:+.4f} | {f['ci95']} |
| Folding TM-score，best-of-5 | {summary['folding']['best5_tm_score']['native']:.4f} | {summary['folding']['best5_tm_score']['omni']:.4f} | {summary['folding']['best5_tm_score']['delta']:+.4f} | {summary['folding']['best5_tm_score']['ci95']} |
| Design 成功率，6 题 × 4 个候选 | {d['native']:.1%} | {d['omni']:.1%} | {d['delta']:+.1%} | {d['ci95']} |

Design 使用每个骨架 8 条 ProteinMPNN 序列及 ESMFold。至少一条同时达到全链 Cα-RMSD≤2Å、motif backbone RMSD≤1Å、pLDDT≥70、平均 pAE≤5 才算成功。失败候选全部计入分母。另报告 TM≥0.6 的单链接去重成功数。

两项点估计均提高：{summary['both_point_estimates_improve']}。两项区间均排除零且为正：{summary['both_ci_exclude_zero_above']}。

Folding 来自作者公开的单体验证名单，序列输入，不含 MSA；不是尚未公开清单的论文 Figure 15 测试集。Design 来自论文 Table 8 的六个 RFdiffusion 任务。小子集只支持本次条件下的结论。Design 指标衡量骨架可设计性，不能单独证明新增序列头更强。

训练曲线及固定样本损失见 W&B 和 runs/native-1729/metrics.jsonl；完整逐题分数，以及全部预测的骨架坐标和序列均保留。
'''
    (ROOT/'RESULTS.md').write_text(report)
    tracker=Tracker(output/'report-tracking',dict(manifest_sha256=file_hash(ROOT/'benchmark_manifest.json'),checkpoint_sha256=summary['checkpoint_sha256'],sources={p.name:file_hash(p) for p in ROOT.glob('*.py')}),'native-condition-poc-report',job_type='evaluation')
    try:
        values={}
        for task in ('folding','design'):
            for metric,record in summary[task].items():
                if isinstance(record,dict) and 'delta' in record:
                    for key in ('native','omni','delta'):values[f'{task}/{metric}/{key}']=record[key]
        values['joint/both_point_estimates_improve']=int(summary['both_point_estimates_improve'])
        values['joint/both_ci_positive']=int(summary['both_ci_exclude_zero_above']);tracker.log(values,400)
        import wandb
        artifact=wandb.Artifact('native-condition-poc-report',type='evaluation')
        for p in [ROOT/'RESULTS.md',ROOT/'benchmark_manifest.json',output/'summary.json',output/'folding_scores.csv',output/'design_scores.csv']:artifact.add_file(str(p))
        tracker.run.log_artifact(artifact)
    finally:tracker.finish(True)
    print('FINAL_RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
