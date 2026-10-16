"""Preserve native generation, with one attempt per fixed candidate slot."""
import importlib.util
import os
import json
from pathlib import Path
from runtime import ROOT, SOURCE, sha, atomic_json

class OneAttemptRecorder:
    def __init__(self, test_set, n_samples, save_dir, **kwargs):
        assert len(test_set) == n_samples == 1
        self.path = Path(save_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.records = []

    def is_finished(self):
        return bool(self.records)

    @property
    def num_generated(self):
        return len(self.records)

    @property
    def num_failed(self):
        return sum(not r['success'] for r in self.records)

    def get_next_batch_list(self, batch_size):
        assert batch_size == 1 and not self.records
        return [(0, 0)]

    def check_and_save(self, log, item_idx, n, **kwargs):
        assert item_idx == n == 0 and not self.records
        record = dict(log or {}, success=log is not None)
        if log is None:
            record['failure_stage'] = kwargs.get('failure_stage', 'native_validity_or_filter')
        self.records.append(record)
        atomic_json(self.path / 'ATTEMPT.json', record)


def native_single_attempt_module():
    source = SOURCE / 'api/helpers/gen_utils.py'
    text = source.read_text()
    assert text.count('for try_i in range(5)') == 1
    text = text.replace('for try_i in range(5)', 'for try_i in range(1)')
    needle = "            likelihoods = [ll.get('likelihood', None) for ll in batch_ll]"
    assert text.count(needle) == 1
    text = text.replace(needle, "                else:\n                    for item_idx, n in batch_list:\n                        recorder.check_and_save(None, item_idx, n, failure_stage='model_sample')\n                    continue\n" + needle)
    path = ROOT / 'runtime/native_single_attempt.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)
    spec = importlib.util.spec_from_file_location('api.helpers.fixedcdr_single_attempt', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Recorder = OneAttemptRecorder
    return module, sha(source), sha(path)

