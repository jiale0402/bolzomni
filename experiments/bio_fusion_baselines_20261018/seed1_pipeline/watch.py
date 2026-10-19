"""Bounded Slurm supervisor. No retries of started training or generation."""
import json
import os
from pathlib import Path
import subprocess
import time
import pipeline

ROOT = Path(__file__).resolve().parent


def main():
    job = os.environ['SLURM_JOB_ID']
    started = time.monotonic()
    while time.monotonic()-started < 3*3600+40*60:
        if any((p/'STOPPED.json').exists() for p in [ROOT, ROOT.parent]):
            return
        if (ROOT/'FULL_COMPLETE.json').exists():
            return
        result = subprocess.run(['python3', str(ROOT/'pipeline.py'), '--advance'], capture_output=True, text=True)
        with (ROOT/'WATCH_EVENTS.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix=time.time(), supervisor_job=job, returncode=result.returncode,
                                    output=result.stdout[-6000:], error=result.stderr[-3000:]))+'\n')
        if (ROOT/'training/FAILED.json').exists():
            return
        # A failed/timed-out execution requires diagnosis, not another paid run.
        status = json.loads((ROOT/'PIPELINE_STATUS.json').read_text()) if (ROOT/'PIPELINE_STATUS.json').exists() else {}
        lines = status.get('slurm', '').splitlines()
        terminal_fail = ('FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL')
        if any(any(t in line for t in terminal_fail) for line in lines):
            return
        time.sleep(50)
    if not (ROOT/'FULL_COMPLETE.json').exists():
        # A new bounded supervisor continues lightweight progression after this
        # allocation using the configured scheduler account.
        receipts = list(ROOT.glob('watch*_SUBMISSION.json'))
        index = len(receipts)
        if index < 5:
            pipeline.submit('watch'+str(index), ['--partition',os.environ.get('CPU_PARTITION','cpu'),'--time','03:55:00',
                '--mem','1G','--cpus','1','--dependency','afterany:'+job], 'watch.sbatch')


if __name__ == '__main__':
    main()
