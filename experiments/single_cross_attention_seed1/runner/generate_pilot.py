"""Verify inference cache on training inputs, then commit the fixed first chunk."""
import functools
import hashlib
import os
from pathlib import Path
import sys
import time
import torch
import generate_chunk as native


def main():
    study = Path(__file__).resolve().parent.parent
    os.environ['FUSION_ROOT'] = str(study)
    sys.path.append(str(study/'code'))
    from common import manifest
    from native_checks import native_loss, fixed_rng
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    model, epsilon = native.load_model()
    model.cuda()
    started = time.monotonic()
    tracker = native.Tracker('inference-cache-check', {'training_updates':0, 'benchmark_attempts':0})
    records = []
    for modality in ('peptide','antibody','ligand'):
        rows = {r['key']:r for r in manifest()['train'][modality] if r['nblocks']<=120}
        key = min(rows,key=lambda k:hashlib.sha256(('gradient-trace:'+k).encode()).hexdigest())
        captured={}
        hook=epsilon.register_forward_pre_hook(lambda _,args:captured.update(args=args))
        try:
            with torch.inference_mode(),fixed_rng(20261010):
                native_loss(model,epsilon,[rows[key]])
        finally:
            hook.remove()
        args=captured['args'];features=epsilon.features
        with torch.inference_mode():
            epsilon.set_features(features,cache=False)
            uncached=epsilon(*args)
            uncached_repeat=epsilon(*args)
            epsilon.set_features(features,cache=True)
            assert epsilon.fusion._donor_cache is not None
            cached=epsilon(*args)
            cached_repeat=epsilon(*args)
            differences=[float((a-b).abs().max()) for a,b in zip(uncached,cached)]
            native_repeat=[float((a-b).abs().max()) for a,b in zip(uncached,uncached_repeat)]
            cache_repeat=[float((a-b).abs().max()) for a,b in zip(cached,cached_repeat)]
            # Native CUDA scatter/reduction is not bitwise reproducible. Check
            # both repeat paths and cache equivalence at explicit float32 error
            # tolerances; retain all observed errors instead of claiming exactness.
            for left,right in [(uncached,uncached_repeat),(cached,cached_repeat),(uncached,cached)]:
                for a,b in zip(left,right):
                    torch.testing.assert_close(a,b,rtol=1e-5,atol=5e-6)
        record={'modality':modality,'train_key':key,'native_H_X_max_difference':differences,
                'uncached_repeat_max_difference':native_repeat,'cached_repeat_max_difference':cache_repeat}
        records.append(record);tracker.log({'records_checked':len(records),'max_difference':max(differences)})
    proof={'passed':True,'exact_native_epsilon_match':all(max(r['native_H_X_max_difference'])==0 for r in records),
           'native_epsilon_close':True,'rtol':1e-5,'atol':5e-6,'records':records,'train_records':3,
           'validation_or_benchmark_records_used':0,'optimizer_updates':0,'generation_attempts':0,
           'seconds':time.monotonic()-started,'cache_source_sha256':native.sha(study/'runner/cached_fusion.py'),
           'checkpoint_sha256':native.CONFIG['checkpoint_sha256'],'gpu':torch.cuda.get_device_name()}
    native.atomic_json(study/'CACHE_GPU_CHECK.json',proof)
    tracker.finish(proof,[study/'CACHE_GPU_CHECK.json'])
    # User-requested loss on real benchmark references, after checkpoint lock.
    # This is evaluation compute; the one-hour training budget is unchanged.
    native.load_model=functools.lru_cache(maxsize=1)(lambda:(model,epsilon))
    os.environ['BENCH_CHUNK_INDEX']='0'
    native.main()


if __name__=='__main__':main()
