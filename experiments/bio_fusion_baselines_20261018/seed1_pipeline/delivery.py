"""Render the committed benchmark and coverage without filtering candidate slots."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    train=json.loads((ROOT/'training/COMPLETE.json').read_text())
    cfg=json.loads((ROOT/'code/config.json').read_text())
    report=json.loads((ROOT/'benchmark/REPORT.json').read_text())
    interface=json.loads((ROOT/'interface_metrics/SUMMARY.json').read_text())
    comparison=json.loads((ROOT/'comparison/RESULTS.json').read_text())
    assert cfg['seed']==1 and train['trainable_parameters']==4027715
    assert train['checkpoint_sha256']==report['checkpoint_sha256']
    e=html.escape
    parts=['<!doctype html><meta charset="utf-8"><title>EnzyAdapter seed 1</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 22px;color:#142331}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}code,pre{background:#f1f5f8;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}h2{margin-top:32px}</style>',
        '<h1>EnzyAdapter · matched capacity · seed 1</h1>',
        f"<p>Frozen Anew; {train['trainable_parameters']:,} trainable fusion parameters. One A100, 14,400-second startup-inclusive training budget.</p>",
        f"<p>Optimizer steps: {train['optimizer_steps']}; presentations: {train['presentations']}. Selected checkpoint: <code>{e(train['checkpoint_sha256'])}</code></p>",
        '<p>One shared checkpoint selected on the fixed 614-record validation set. The previously inspected 253-target benchmark holdout does not select weights. Known ligand training-PDB overlaps: 19.</p>',
        '<p>253 targets × 30 fixed attempted slots. Failed attempts and missing metric values are retained. These are component-transfer results, not a reproduction of the source-domain EnzyControl model.</p>',
        '<p>Energy values are computational Rosetta REU, not measured binding affinity. Fixed time and matched trainable parameter count do not establish matched FLOPs. Read the per-metric finite denominators.</p>']
    total=0
    for name,value in report['benchmark_results'].items():
        all_=value['all'];total+=all_['attempted']
        parts += [f'<h2>{e(name)}</h2>',f"<p>Attempted {all_['attempted']}; generated {all_['generated']}; candidates with metric errors {all_['metric_error_candidates']}.</p>",
            '<table><tr><th>Native metric (raw units)</th><th>Mean</th><th>Median</th><th>Finite candidates</th></tr>']
        for metric,stats in all_['metrics'].items():
            parts.append(f"<tr><td>{e(metric)}</td><td>{stats['mean']:.6g}</td><td>{stats['median']:.6g}</td><td>{stats['count']}</td></tr>")
        parts.append('</table>')
    assert total==7590
    parts += ['<h2>Protein interface metrics and coverage</h2><p>153 protein targets. DockQ/iRMSD/fnat and backbone interface lDDT retain their own finite denominators and the unchanged scoring regions.</p>',
        '<pre>'+e(json.dumps(interface,indent=2))+'</pre>',
        '<h2>Validation and benchmark-holdout H/X/total</h2><pre>'+e(json.dumps({'validation':comparison['validation'],'holdout':comparison['benchmark_holdout_loss']},indent=2))+'</pre>',
        '<h2>Paired comparison against the historical method</h2><pre>'+e((ROOT/'comparison/RESULTS.md').read_text())+'</pre>',
        '<p>W&amp;B online/offline lifecycle receipts are in FULL_COMPLETE.json. Server-side finished/COMMITTED verification and final Slurm accounting are checked separately before final delivery.</p>']
    dest=ROOT/'delivery';dest.mkdir(exist_ok=True)
    (dest/'RESULTS.html').write_text('\n'.join(parts)+'\n')
    (dest/'RESULTS.json').write_text(json.dumps({'training':train,'benchmark':report,'interface':interface,'comparison':comparison},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
