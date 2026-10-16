"""Unpack pinned source snapshots, fetch fixed model assets, and register environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
BASE_URL = 'https://github.com/bytedance/AnewOmni/releases/download/init/model.ckpt'
ASSETS = {
    'boltzgen1_diverse.ckpt': ('boltzgen/boltzgen-1', 'model', '360af8bd6e59527ff6ec25dd81253967f3bd3567d200053b10680634751f8e3c'),
    'mols.zip': ('boltzgen/inference-data', 'dataset', '3d4f56ac4262e745bb3d09cfaa19099b1d01be208122d501667b952e45521e53'),
}
BASE_SHA = '961599acb617b8baf742755b4416322979507c9814ebd1a1e8582cff9c0a74c1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_sources():
    spec = json.loads((HERE / 'SOURCE_INDEX.json').read_text())
    for entry in spec['archives']:
        archive = HERE / entry['archive']
        assert sha(archive) == entry['sha256'], archive
        expected = {x['path']: x for x in entry['members']}
        seen = set()
        with tarfile.open(archive, 'r:gz') as tf:
            for member in tf:
                name = PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts
                assert member.name in expected and member.name not in seen
                item = expected[member.name]
                assert member.size == item['bytes']
                dest = (HERE / member.name).resolve()
                assert dest.is_relative_to(HERE) and member.name.startswith('vendor/')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.is_file() or sha(dest) != item['sha256']:
                    with tf.extractfile(member) as source, dest.open('wb') as output:
                        shutil.copyfileobj(source, output)
                assert sha(dest) == item['sha256']
                seen.add(member.name)
        assert seen == set(expected)


def executable(value):
    path = shutil.which(value) or value
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Do not resolve a venv interpreter symlink: its environment prefix matters.
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-python')
    parser.add_argument('--donor-python')
    parser.add_argument('--score-python')
    parser.add_argument('--donor-extra-pythonpath', type=Path, action='append', default=[])
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--boltzgen-assets', type=Path)
    parser.add_argument('--assets-dir', type=Path, default=HERE / 'assets')
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE / 'runtime.json')
    args = parser.parse_args()
    extract_sources()
    if args.sources_only:
        print('SOURCE_SNAPSHOTS_VERIFIED', HERE / 'vendor')
        return
    if not all([args.generation_python, args.donor_python, args.score_python]):
        parser.error('Provide the three installed environment Python paths (see README.md)')
    assets = args.assets_dir.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    base = args.base_checkpoint.resolve() if args.base_checkpoint else assets / 'anewomni_official.ckpt'
    if not base.is_file():
        if args.base_checkpoint:
            raise FileNotFoundError(base)
        temporary = base.with_suffix('.download')
        with urllib.request.urlopen(BASE_URL, timeout=120) as src, temporary.open('wb') as out:
            shutil.copyfileobj(src, out, 8 * 1024 * 1024)
        assert sha(temporary) == BASE_SHA
        temporary.replace(base)
    assert sha(base) == BASE_SHA, 'Fixed Anew base hash mismatch'
    donor_assets = args.boltzgen_assets.resolve() if args.boltzgen_assets else assets / 'boltzgen'
    donor_assets.mkdir(parents=True, exist_ok=True)
    for name, (repo, kind, digest) in ASSETS.items():
        target = donor_assets / name
        if not target.is_file():
            from huggingface_hub import hf_hub_download
            downloaded = Path(hf_hub_download(repo, name, repo_type=kind))
            assert sha(downloaded) == digest, name
            shutil.copyfile(downloaded, target)
        assert sha(target) == digest, name
    runtime = {'generation_python': executable(args.generation_python),
               'donor_python': executable(args.donor_python), 'score_python': executable(args.score_python),
               'anew_source': str(HERE / 'vendor/AnewOmni'),
               'boltzgen_source': str(HERE / 'vendor/boltzgen'),
               'pepglad_source': str(HERE / 'vendor/PepGLAD'), 'cbg_source': str(HERE / 'vendor/CBGBench'),
               'base_checkpoint': str(base), 'boltzgen_assets': str(donor_assets),
               'source_index_sha256': sha(HERE / 'SOURCE_INDEX.json')}
    if args.donor_extra_pythonpath:
        runtime['donor_extra_pythonpath'] = [str(p.resolve()) for p in args.donor_extra_pythonpath]
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runtime, indent=2) + '\n')
    print('RUNTIME_REGISTERED', args.output.resolve())


if __name__ == '__main__':
    main()
