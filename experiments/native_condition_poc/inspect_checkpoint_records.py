"""Read saved checkpoint bytes/optimizer step records; never import or run a model.

Uses a restricted pickle metadata reader, not torch.load. Compares parameter
storage bytes, not predictions, losses, gradients or numerical tensor diagnostics.
"""
import collections
import hashlib
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path


class Placeholder:
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def tensor(storage, offset, size, stride, *args):
    return dict(storage=storage, offset=offset, shape=size, stride=stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) == ('_codecs', 'encode'):
            return lambda value, encoding: value.encode(encoding)
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_tensor_v3'):
            return tensor
        if module in ('numpy', 'numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'ndarray', 'dtype'):
            return Placeholder
        raise ValueError('Unsupported checkpoint metadata type: ' + module + '.' + name)

    def persistent_load(self, key):
        assert key[0] == 'storage'
        return dict(dtype=key[1], key=key[2], device=key[3], count=key[4])


def open_records(path):
    archive = zipfile.ZipFile(path)
    entry = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
    root = entry.rsplit('/', 1)[0]
    record = MetadataReader(io.BytesIO(archive.read(entry))).load()
    return archive, root, record


def storage_hash(archive, root, desc):
    digest = hashlib.sha256()
    with archive.open(root + '/data/' + str(desc['storage']['key'])) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scalar(archive, root, desc):
    assert desc['shape'] == () and desc['offset'] == 0
    formats = {'FloatStorage': '<f', 'DoubleStorage': '<d', 'LongStorage': '<q'}
    return struct.unpack(formats[desc['storage']['dtype']], archive.read(root + '/data/' + str(desc['storage']['key'])))[0]


def main():
    root = Path('/projects/p53252/jck3114/boltz_omni_poc/experiments/native_condition_poc/runs/native-1729')
    initial, ri, a = open_records(root / 'initial.pt')
    final, rf, b = open_records(root / 'final.pt')
    resume, rr, c = open_records(root / 'resume.pt')
    assert a['step'] == 0 and b['step'] == c['step'] == 400
    names = list(a['adapter'])
    assert set(names) == set(b['adapter']) == set(c['adapter'])
    grouped_names = [[n for n in names if n.startswith(prefix)] for prefix in ('base.', 'adapter.')]
    assert sum(map(len, grouped_names)) == len(names)
    rows = []
    for group, group_names in zip(c['optimizer']['param_groups'], grouped_names):
        assert len(group['params']) == len(group_names)
        for name, pid in zip(group_names, group['params']):
            before, after = a['adapter'][name], b['adapter'][name]
            assert before['shape'] == after['shape'] and before['offset'] == after['offset'] == 0
            state = c['optimizer']['state'].get(pid)
            rows.append(dict(name=name, shape=after['shape'], dtype=after['storage']['dtype'],
                initial_storage_sha256=storage_hash(initial, ri, before),
                final_storage_sha256=storage_hash(final, rf, after),
                optimizer_state_present=state is not None,
                optimizer_step=scalar(resume, rr, state['step']) if state else None,
                final_lr=group['lr']))
    groups = {}
    for prefix in ('base.structure_module.', 'base.diffusion_conditioning.', 'adapter.'):
        chosen = [r for r in rows if r['name'].startswith(prefix)]
        groups[prefix] = dict(parameter_tensors=len(chosen),
            changed_storage_count=sum(r['initial_storage_sha256'] != r['final_storage_sha256'] for r in chosen),
            optimizer_steps=dict(collections.Counter(str(r['optimizer_step']) for r in chosen)),
            dtypes=dict(collections.Counter(r['dtype'] for r in chosen)),
            unchanged_names=[r['name'] for r in chosen if r['initial_storage_sha256'] == r['final_storage_sha256']])
    print(json.dumps(dict(scope='Read existing checkpoint metadata, optimizer scalar records and parameter storage byte hashes only; no torch, numpy, model execution or tensor arithmetic.',
        initial_step=a['step'], final_step=b['step'], resume_step=c['step'], groups=groups, parameters=rows), indent=2))


if __name__ == '__main__':
    main()
