"""Deterministic metadata-only selection from the released BoltzGen training archive."""
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from remote_zip import open_archive, REVISION, URL

ROOT = Path(__file__).resolve().parent
UPSTREAM = 'a3149cf18eeb58648d1abbb27539bd73f746cdda'


def records(stream):
    decoder = json.JSONDecoder()
    buffer = ''
    end = False
    while True:
        buffer = buffer.lstrip(' \r\n\t[,')
        if buffer.startswith(']'): return
        try:
            row, pos = decoder.raw_decode(buffer)
            buffer = buffer[pos:]
            yield row
        except json.JSONDecodeError:
            if end: raise
            chunk = stream.read(4*1024*1024)
            end = not chunk
            buffer += chunk


def main():
    root = ROOT/'data'; root.mkdir(exist_ok=True)
    split = root/'validation_ids_boltz2_all.txt'
    if not split.exists():
        split.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/HannesStark/boltzgen/{UPSTREAM}/src/boltzgen/resources/splits/validation_ids_boltz2_all.txt').read())
    excluded = {s.strip().lower() for s in split.read_text().splitlines()}
    candidates_path = root/'candidates.json'
    if candidates_path.exists(): return
    source, archive = open_archive(root/'ranges')
    training, testing = [], []
    with archive.open('targets/manifest.json') as binary:
        for i, row in enumerate(records(io.TextIOWrapper(binary, encoding='utf-8'))):
            chains = row['chains']
            if len(chains) != 1 or chains[0]['mol_type'] != 0 or not chains[0]['valid']: continue
            length = chains[0]['num_residues']
            if not 60 <= length <= 250: continue
            meta = row['structure']; date = meta.get('released') or ''
            resolution = meta.get('resolution')
            if resolution is None or resolution <= 0 or resolution >= 4.5: continue
            row = {**row, 'selection_sha256': hashlib.sha256(('1729:'+row['id']).encode()).hexdigest()}
            if date and date <= '2023-06-01' and row['id'].lower() not in excluded and 80 <= length <= 160:
                training.append(row)
            if row['id'].lower() in excluded and '2024-01-01' <= date <= '2024-12-31':
                testing.append(row)
    training.sort(key=lambda x:x['selection_sha256']); testing.sort(key=lambda x:x['selection_sha256'])
    candidates_path.write_text(json.dumps(dict(training=training, testing=testing, dataset_revision=REVISION,
        archive_url=URL, split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        rules='Full single-protein assemblies; train 80-160aa, test 60-250aa; resolution (0,4.5); official release cutoff/split',
        bytes_downloaded=source.downloaded), indent=2))
    print('OFFICIAL_CANDIDATES', len(training), len(testing), flush=True)


if __name__ == '__main__': main()
