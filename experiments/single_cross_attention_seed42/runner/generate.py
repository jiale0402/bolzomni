"""Fixed candidate slots with resumable batch receipts and native generation."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from runtime import CONFIG, ROOT, SOURCE, CACHE_ROOT
sys.path.insert(0, str(SOURCE))
import torch
from runtime import load_model, sha, seed, Tracker, atomic_json
from fusion import Fusion, FusionEpsilon
from benchmark_inputs import dataset
from feature_contract import bind_features
from export_audit import audit_export
from native_generator import native_single_attempt_module


def slot_seed(benchmark, target, slot):
    return int.from_bytes(hashlib.sha256(f"{CONFIG['seed']}:{benchmark}:{target}:{slot}".encode()).digest()[:4], 'big')


class BatchRecorder:
    def __init__(self, test_set, n_samples, save_dir, **kwargs):
        assert len(test_set) == 1
        self.count = n_samples
        self.path = Path(save_dir)
        self.records = {}
        assert not (self.path / 'attempts.jsonl').exists(), 'Inspect incomplete batch before resuming'

    def is_finished(self):
        return len(self.records) == self.count

    @property
    def num_generated(self):
        return len(self.records)

    @property
    def num_failed(self):
        return sum(not x['success'] for x in self.records.values())

    def get_next_batch_list(self, batch_size):
        assert batch_size == self.count and not self.records
        return [(0, n) for n in range(self.count)]

    def check_and_save(self, log, item_idx, n, **kwargs):
        assert item_idx == 0 and 0 <= n < self.count and n not in self.records
        record = dict(log or {}, success=log is not None, n=n)
        if log is None:
            record['failure_stage'] = kwargs.get('failure_stage', 'native_validity_or_filter')
        self.records[n] = record
        with (self.path / 'attempts.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()


def bind_control(feature, cached, current, benchmark):
    from feature_contract import context_positions
    assert torch.equal(feature['generate_mask'], cached['generate_mask'])
    if benchmark != 'crossdocked2020':
        assert torch.equal(cached['generate_mask'], current['generate_mask'])
        return feature
    assert context_positions(cached) == context_positions(current)
    assert not feature['mask'][cached['generate_mask']].any()
    mask = torch.zeros_like(current['generate_mask'])
    mask[~current['generate_mask']] = feature['mask'][~cached['generate_mask']]
    return {'mask': mask, 'generate_mask': current['generate_mask']}


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for benchmark generation')
    cache = CACHE_ROOT
    rows = json.loads((cache / 'manifests/feature_rows.json').read_text())
    rows.sort(key=lambda r: (int(r['target'].split('_')[1]), r['benchmark']))
    model, epsilon = load_model()
    model.cuda()
    args = argparse.Namespace(batch_size=CONFIG['batch_size'], method='omni', shard=0)
    selection = {'sha256': CONFIG['checkpoint_sha256']}
    policy = {'benchmarks': CONFIG['benchmark_specs']}
    ev = ROOT
    functions, native_hash, runtime_hash = native_single_attempt_module()
    functions.Recorder = BatchRecorder
    provenance = {'shared_checkpoint_sha256': selection['sha256'], 'method': args.method,
                  'gamma': CONFIG['gamma'], 'batch_size': args.batch_size,
                  'native_generator_sha256': native_hash, 'runtime_sha256': runtime_hash,
                  'run_hash': CONFIG['run_hash'], 'source_hashes': CONFIG['source_hashes']}
    tracker = Tracker(f'evaluation-{args.method}-shard{args.shard}', provenance)
    started = time.monotonic(); finished = successful = 0
    for row in rows:
        benchmark, target_id, key = row['benchmark'], row['target'], row['key']
        spec = next(x for x in policy['benchmarks'] if x['name'] == benchmark)
        assert sha(spec['manifest']) == spec['manifest_sha256']
        target = next(json.loads(x) for x in Path(spec['manifest']).read_text().splitlines() if json.loads(x)['id'] == target_id)
        for name, digest in row['files'].items():
            assert sha(cache / name) == digest
        cached = torch.load(cache / 'inputs' / key / 'api_input.pt', map_location='cpu', weights_only=False)
        ds = dataset(benchmark, target)
        feature = None
        if CONFIG['variant'] == 'full':
            fp = cache / 'features/aligned' / (key + '.pt')
            status = json.loads((cache / 'features/status' / (key + '.json')).read_text())
            assert status['sha256'] == sha(fp) and status['input_sha256'] == row['input_sha256']
            feature = torch.load(fp, map_location='cpu', weights_only=True)
        if CONFIG['variant'] == 'control':
            meta = json.loads((cache / 'inputs' / key / 'metadata.json').read_text())
            feature = {'mask': torch.tensor([x is not None for x in meta['mapping']]),
                       'generate_mask': torch.tensor(meta['generate_mask'])}
        output = ev / 'generation'  / args.method / benchmark
        for first in range(0, CONFIG['samples'], args.batch_size):
            slots = list(range(first, min(first + args.batch_size, CONFIG['samples'])))
            work = output / 'transactions' / target_id / f'batch-{first:02d}'
            work.mkdir(parents=True, exist_ok=True)
            receipt = work / 'COMMITTED.json'
            if receipt.exists():
                batch = json.loads(receipt.read_text())
                assert batch['provenance'] == provenance and batch['slots'] == slots
                for name, digest in batch['files'].items():
                    assert sha(work / name) == digest
            else:
                assert not (work / 'STARTED.json').exists(), 'Uncommitted batch requires diagnosis; do not repeat attempts'
                prepared = []
                for n in slots:
                    seed(slot_seed(benchmark, target_id, n))
                    data = ds[0] if benchmark == 'crossdocked2020' else copy.deepcopy(cached)
                    assert data['cplx_desc'].id == target_id
                    prepared.append(data)
                class CachedBatch(type(ds)):
                    def __getitem__(self, index):
                        assert index == 0 and self.cursor < len(prepared)
                        result = copy.deepcopy(prepared[self.cursor]); self.cursor += 1
                        return result
                batch_ds = copy.copy(ds); batch_ds.__class__ = CachedBatch; batch_ds.cursor = 0
                if epsilon is not None:
                    binder = bind_features if CONFIG['variant'] == 'full' else bind_control
                    features = [binder(feature, cached, data, benchmark) for data in prepared]
                    epsilon.set_features([{k: v.cuda() for k, v in f.items()} for f in features])
                seed(slot_seed(benchmark, target_id, first))
                atomic_json(work / 'STARTED.json', {'slots': slots, 'provenance': provenance})
                tick = time.monotonic()
                functions.generate_for_one_template(model, batch_ds, len(slots), len(slots), str(work),
                    torch.device('cuda'), {'cal_likelihood': False}, n_cycles=1, max_retry=1, verbose=False)
                records = [json.loads(x) for x in (work / 'attempts.jsonl').read_text().splitlines()]
                assert sorted(x['n'] for x in records) == list(range(len(slots)))
                records.sort(key=lambda x: x['n'])
                for local, record in enumerate(records):
                    if record['success']:
                        record['export_audit'] = audit_export(prepared[local], record, work / target_id / f'{local}.pdb')
                files = {str(p.relative_to(work)): sha(p) for record in records if record['success']
                         for p in (work / target_id).glob(str(record['n']) + '.*')}
                batch = {'slots': slots, 'records': records, 'provenance': provenance, 'files': files,
                         'seconds': time.monotonic() - tick}
                atomic_json(receipt, batch)
            for local, n in enumerate(slots):
                record = dict(batch['records'][local], id=target_id, n=n, benchmark=benchmark)
                dest = output / target_id; dest.mkdir(parents=True, exist_ok=True)
                if record['success']:
                    for src in (work / target_id).glob(f'{local}.*'):
                        shutil.copyfile(src, dest / (str(n) + src.suffix))
                    record['pdb_sha256'] = sha(dest / f'{n}.pdb')
                atomic_json(output / 'slots' / target_id / f'{n:02d}.json', record)
                finished += 1; successful += int(record['success'])
            tracker.log({'coverage/attempted': finished, 'coverage/successful': successful,
                         'system/elapsed_seconds': time.monotonic() - started})
    report = {'attempted': finished, 'successful': successful, 'expected': len(rows) * CONFIG['samples'],
              'seconds': time.monotonic() - started, 'provenance': provenance}
    assert finished == report['expected']
    path = ev / 'generation' / args.method / f'COMPLETE-shard{args.shard}.json'
    atomic_json(path, report); tracker.finish(report, [path, Path(CONFIG['checkpoint'])])


if __name__ == '__main__':
    main()
