"""Independent W&B stream per disjoint benchmark chunk; generation unchanged."""
import os
import generate_chunk as native


class ChunkTracker(native.Tracker):
    def __init__(self,stage,config=None):
        super().__init__(stage+'-chunk'+os.environ['BENCH_CHUNK_INDEX'],config)


native.Tracker=ChunkTracker
native.main()
