"""Native generation, with separate W&B streams for the two hosts."""
import os
import generate_chunk as native


class HostTracker(native.Tracker):
    def __init__(self, stage, config=None):
        super().__init__(stage + '-' + os.environ['BENCH_HOST'], dict(config or {},
            generation_scope='host-local chunk stream, not full benchmark completion',
            generation_host=os.environ['BENCH_HOST']))

    def log(self, values):
        values = dict(values)
        values['chunk/index'] = int(os.environ['BENCH_CHUNK_INDEX'])
        for key in ['coverage/attempted', 'coverage/successful']:
            if key in values:
                values['native_chunk_stream/' + key.split('/')[-1]] = values.pop(key)
        super().log(values)


native.Tracker = HostTracker
native.main()
