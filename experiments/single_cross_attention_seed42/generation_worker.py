"""Four disjoint chunk workers; native sampling and fixed slots are unchanged."""
import functools
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runner'))
import generate_chunk as native


def main():
    worker=int(os.environ['BENCH_WORKER_INDEX'])
    assert worker in range(4)
    native.load_model=functools.lru_cache(maxsize=1)(native.load_model)
    old_tracker=native.Tracker
    class ChunkTracker(old_tracker):
        def __init__(self,stage,config=None):
            super().__init__(stage+'-chunk'+os.environ['BENCH_CHUNK_INDEX'],config)
    native.Tracker=ChunkTracker
    for chunk in range(worker,31,4):
        receipt=native.ROOT/'generation/omni'/f'COMPLETE-chunk{chunk:03d}.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            assert saved['provenance']['run_hash']==native.CONFIG['run_hash']
            assert saved['attempted']==saved['expected']
            continue
        os.environ['BENCH_CHUNK_INDEX']=str(chunk)
        native.main()
    native.atomic_json(ROOT/f'GENERATION_WORKER_{worker}.json',{'complete':True,'worker':worker})


if __name__=='__main__':main()
