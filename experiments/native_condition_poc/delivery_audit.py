"""CPU-only inspection of saved parameters and final evidence publication."""
import json
import subprocess
from pathlib import Path
import torch
import wandb
from tracking import Tracker,file_hash

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
RUN=ROOT/'runs/native-1729'
WB='jialechen2025-northwestern-university/boltz-omni-joint'

def main():
    torch.set_num_threads(4)
    initial=torch.load(RUN/'initial.pt',map_location='cpu',weights_only=False,mmap=True)['adapter']
    final=torch.load(RUN/'final.pt',map_location='cpu',weights_only=False,mmap=True)['adapter']
    assert set(initial)==set(final)
    projections={}
    for key in ('adapter.token_out.weight','adapter.pair_out.weight'):
        a,b=initial[key].float(),final[key].float()
        projections[key]=dict(initial_norm=float(a.norm()),final_norm=float(b.norm()),
            changed_parameters=int((a!=b).sum()),parameter_count=b.numel())
        assert torch.isfinite(b).all() and projections[key]['changed_parameters']>0
    head=initial['adapter.identity_readout.1.weight']
    head_delta=float((final['adapter.identity_readout.1.weight']-head).float().norm())
    assert head_delta>0
    assert all(k.startswith(('base.structure_module.','base.diffusion_conditioning.','adapter.')) for k in final)
    mechanism=dict(projections=projections,identity_head_parameter_delta_norm=head_delta,
        final_checkpoint_sha256=file_hash(RUN/'final.pt'),source_sha256=file_hash(__file__),
        inference_source_sha256=file_hash(ROOT/'native_flow.py'),inspection_only_no_model_execution=True)
    (OUT/'mechanism_audit.json').write_text(json.dumps(mechanism,indent=2))
    del initial,final
    accounting=json.loads((OUT/'completion_audit.json').read_text())
    ids=[json.loads(p.read_text())['job_id'].split(';')[0] for p in (ROOT/'submissions').glob('*.json')]
    raw=subprocess.check_output(['sacct','-nP','-j',','.join(ids),
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,Account'],text=True)
    records=[];gpu_hours=0.
    for line in raw.splitlines():
        p=line.split('|')
        if p[0] not in ids:continue
        gpu=next((int(x.split('=')[1]) for x in p[4].split(',') if x.startswith('gres/gpu=')),0)
        if gpu:assert p[1] not in ('RUNNING','PENDING','COMPLETING')
        gpu_hours+=gpu*int(p[3])/3600
        records.append(dict(job=p[0],state=p[1],exit_code=p[2],seconds=int(p[3]),gpu_count=gpu,account=p[5]))
    assert abs(gpu_hours-accounting['allocated_GPU_hours'])<1e-9
    generated=[json.loads(p.read_text()) for p in OUT.glob('*-omni.json')]
    assert len(generated)==64
    geom=dict(finite=sum(r['finite_geometry'] for r in generated),total=64,
        largest_CA_gap_A=max(r['max_ca_distance'] for r in generated),
        total_abnormal_CA_gaps=sum(r['abnormal_ca_distances'] for r in generated))
    summary=json.loads((OUT/'summary.json').read_text())
    rmsd=summary['design']['best_sc_rmsd_mean']
    before=json.loads((RUN/'validation-0.json').read_text())
    after=json.loads((RUN/'validation-400.json').read_text())
    def ce(rows,split):
        vals=[r['identity'] for r in rows if r['split']==split and r['mode']=='design']
        return sum(vals)/len(vals)
    text=(ROOT/'RESULTS.md').read_text()
    marker='\n<!-- delivery-verification -->'
    text=text.split(marker)[0]+marker+f'''

反馈模块实际更新已核验：token 和 pair 两个输出投影的权重均发生变化，序列头也发生更新。
最终 64 次完整生成中 {geom['finite']}/64 坐标有限，最大相邻 Cα 间距 {geom['largest_CA_gap_A']:.3f} Å。
完整生成的连续性检查与 design 成功率是不同指标。

Design 连续指标：各骨架 best-of-8 CA scRMSD 的均值，原版 {rmsd['native']:.4f} Å，
Omni {rmsd['omni']:.4f} Å；差值 {rmsd['delta']:+.4f} Å，任务级配对 95% 区间 {rmsd['ci95']}。
越低越好；该 best-of-8 RMSD 指标不替代同一序列满足全部四项阈值的成功标准。

固定 train/validation design 序列 CE：训练 4 条 {ce(before,'train'):.4f} → {ce(after,'train'):.4f}；
验证 16 条 {ce(before,'validation'):.4f} → {ce(after,'validation'):.4f}。结构损失变化见下图。

本轮实际分配 GPU 时间为 {gpu_hours:.4f} 小时，包含本轮提交记录中的失败尝试；
复用原版历史结果的计算不算作本轮新增用量。原版生成来自之前的 A100，本轮生成使用 H100；
精度配置、采样设置和种子保持原协议。跨 GPU 的数值变化是额外限制，极小差异不能单独归因于训练。

![Training and fixed-probe curves](results/learning_curves.png)
'''
    (ROOT/'RESULTS.md').write_text(text)
    api=wandb.Api(timeout=45);verified=[]
    for d in (RUN,OUT/'generation-tracking',OUT/'refold-tracking',OUT/'report-tracking',OUT/'audit-tracking'):
        status=json.loads((d/'tracking_status.json').read_text());r=api.run(WB+'/'+status['id'])
        assert status['mode']=='online' and r.state=='finished'
        verified.append(dict(id=r.id,state=r.state,url=r.url))
        if d==RUN:
            artifacts=list(r.logged_artifacts())
            assert any(a.type=='model' and a.metadata.get('sha256')==mechanism['final_checkpoint_sha256']
                and a.state=='COMMITTED' for a in artifacts)
    delivery=dict(complete=True,mechanism_verified=True,geometry=geom,wandb_verified=verified,allocated_GPU_hours=gpu_hours,
        jobs=records,results_sha256=file_hash(ROOT/'RESULTS.md'),summary_sha256=file_hash(OUT/'summary.json'),
        learning_curves_sha256=file_hash(OUT/'learning_curves.png'))
    (OUT/'delivery_verification.json').write_text(json.dumps(delivery,indent=2))
    tracker=Tracker(OUT/'delivery-tracking',dict(source_sha256=file_hash(__file__),
        checkpoint_sha256=mechanism['final_checkpoint_sha256']),'native-condition-delivery',job_type='evaluation')
    ok=False
    try:
        tracker.log({'audit/complete':1,'audit/feedback_parameters_updated':1,
            'compute/allocated_GPU_hours_including_failures':gpu_hours},400)
        tracker.run.log({'fixed_probe_and_training_curves':wandb.Image(str(OUT/'learning_curves.png'))})
        artifact=wandb.Artifact('native-condition-delivery',type='evaluation')
        for p in (OUT/'delivery_verification.json',OUT/'mechanism_audit.json',OUT/'learning_curves.png',ROOT/'RESULTS.md'):
            artifact.add_file(str(p))
        tracker.run.log_artifact(artifact);ok=True
    finally:tracker.finish(ok)
    print('DELIVERY_VERIFIED',json.dumps(delivery),flush=True)

if __name__=='__main__':main()
