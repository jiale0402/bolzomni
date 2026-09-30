"""Dependency compatibility only; preserve all benchmark scoring functions."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
logpath=root/'compatibility_changes.json'
changes=json.loads(logpath.read_text()) if logpath.exists() else []
def patch(path, fn):
    text=path.read_text()
    after=fn(text)
    if text==after:return
    original=path.with_suffix(path.suffix+'.original')
    if not original.exists(): original.write_text(text)
    path.write_text(after)
    changes.append(dict(path=str(path),before_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        after_sha256=hashlib.sha256(after.encode()).hexdigest()))

# PyMOL is only needed for optional visualization, never for refolding or RMSD.
for relative in ('analysis/utils.py','analysis/plot.py'):
    patch(root/'vendor/MotifBench/Scaffold-Lab'/relative,
          lambda s:s if 'cmd = None' in s else s.replace('from pymol import cmd','try:\n    from pymol import cmd\nexcept ImportError:\n    cmd = None  # Optional visualization disabled in this evaluation'))

# Python 3.11 rejects mutable dataclass defaults used by fair-esm 2.0.0.
esm=Path(importlib.util.find_spec('esm').origin).parent
for name in ('esmfold/v1/esmfold.py','esmfold/v1/trunk.py'):
    p=esm/name
    def convert(s):
        s=s.replace('from dataclasses import dataclass\n','from dataclasses import dataclass, field\n')
        return re.sub(r'^(\s+\w+:\s*[\w.]+\s*=\s*)(\w+Config)\(\)\s*$',
                      r'\1field(default_factory=\2)',s,flags=re.M)
    patch(p,convert)
# ESMFold uses the ordinary attention path; the optional custom CUDA kernel is
# not built here. Keep it unavailable (and fail if requested), never substitute
# its output. This merely makes the unused extension an optional import.
for relative in ('utils/kernel/attention_core.py','model/structure_module.py'):
    patch(root/'vendor/openfold/openfold'/relative,
          lambda s:s if 'attn_core_inplace_cuda = None' in s else s.replace(
              'attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")',
              'try:\n    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")\nexcept ModuleNotFoundError:\n    attn_core_inplace_cuda = None'))
patch(root/'vendor/openfold/openfold/utils/seed.py',
      lambda s:s.replace('from pytorch_lightning.utilities.seed import seed_everything',
                         'from pytorch_lightning import seed_everything'))
logpath.write_text(json.dumps(changes,indent=2))
print('COMPATIBILITY_PATCHES',len(changes),flush=True)
