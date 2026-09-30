"""Lock existing inputs and copy the already audited, unchanged original baseline."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'official_poc'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not (ROOT/'runs/native-1729/started_updates.json').exists()
    manifest=json.loads((OLD/'benchmark_manifest.json').read_text())
    manifest['checkpoint']='experiments/native_condition_poc/runs/native-1729/final.pt'
    manifest['panel_status']='Previously inspected exploratory repair/regression panel; no test-set novelty claimed.'
    (ROOT/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    evidence=[]
    paths=list((OLD/'results').glob('*-native.json'))+list((OLD/'results').glob('*-native.npz'))
    for subdir in ('folding/native','scaffolds/native','refold/native'):
        paths += [p for p in (OLD/'results'/subdir).rglob('*') if p.is_file()]
    for src in paths:
        rel=src.relative_to(OLD/'results');dest=out/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        assert digest(src)==digest(dest)
        evidence.append(dict(path=str(rel),sha256=digest(src)))
    assert len(list(out.glob('*-native.json')))==64
    assert len(list((out/'refold/native').glob('*/sequence_scores.csv')))==6
    data=json.loads((OLD/'data_manifest.json').read_text())
    cache=[]
    for split in ('train','validation'):
        for row in data[split]:
            for mode in ('folding','design'):
                p=OLD/'cache'/f'{row["id"]}-{mode}.pt'
                # Missing validation caches can be prepared by the training loader.
                cache.append(dict(id=row['id'],mode=mode,split=split,exists=p.exists()))
    lock=dict(baseline=evidence,old_manifest_sha256=digest(OLD/'benchmark_manifest.json'),
        manifest_sha256=digest(ROOT/'benchmark_manifest.json'),data_manifest_sha256=digest(OLD/'data_manifest.json'),
        baseline_audit_sha256=digest(OLD/'results/final_audit.json'),cache_inventory=cache)
    (ROOT/'input_lock.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(dict(baseline_files=len(evidence),cache_files=sum(r['exists'] for r in cache),
        manifest_sha256=lock['manifest_sha256'])))

if __name__=='__main__':main()
