"""Reuse the same frozen model across disjoint chunks; all native draws unchanged."""
import functools
import os
import time
import generate_chunk as native

worker = int(os.environ['BENCH_WORKER_INDEX'])
assert worker in (0, 1, 2, 3)
chunks = list(range(1 + worker, 31, 4))
# Chunk 0 is the independent pilot. Its job must finish before these two workers.
pilot = native.ROOT/'generation/omni/COMPLETE-chunk000.json'
assert pilot.is_file()
assert native.json.loads(pilot.read_text())['provenance']['run_hash'] == native.CONFIG['run_hash']
original_load_model = native.load_model
native.load_model = functools.lru_cache(maxsize=1)(original_load_model)
original_tracker = native.Tracker


class ChunkTracker(original_tracker):
    def __init__(self, stage, config=None):
        super().__init__(stage+'-chunk'+os.environ['BENCH_CHUNK_INDEX'], config)


native.Tracker = ChunkTracker
status_path = native.ROOT/f'GENERATION_WORKER_{worker}.json'
started = time.time()
done = []
try:
    for chunk in chunks:
        os.environ['BENCH_CHUNK_INDEX'] = str(chunk)
        native.atomic_json(status_path, {'pid':os.getpid(), 'stage':'running',
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'), 'worker':worker,
            'chunk':chunk, 'completed_chunks':done, 'assigned_chunks':chunks,
            'unix':time.time(), 'started_unix':started})
        native.main()
        done.append(chunk)
    native.atomic_json(status_path, {'pid':os.getpid(), 'stage':'complete',
        'worker':worker, 'completed_chunks':done, 'assigned_chunks':chunks,
        'unix':time.time(), 'started_unix':started,
        'model_load_count':native.load_model.cache_info().misses})
except BaseException as exc:
    native.atomic_json(status_path, {'pid':os.getpid(), 'stage':'failed',
        'worker':worker, 'completed_chunks':done, 'assigned_chunks':chunks,
        'unix':time.time(), 'error':repr(exc),
        'policy':'Inspect any uncommitted transaction before resuming; do not repeat attempts blindly.'})
    raise
