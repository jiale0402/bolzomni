# Capacity-matched biological fusion baselines

EnzyAdapter and SES-Adapter fusion operators transferred into frozen Anew EPT,
using cached sanitized PairFormer features. These are component transfers, not
reproductions of the source papers' complete models or tasks.

| Module | Effective trainable parameters | Difference from4,027,712 |
| --- | ---: | ---: |
| EnzyAdapter | 4,027,715 | +3 |
| SES-Adapter | 4,009,680 | -0.447698% |

Tolerance is0.5%. SES's892,080 unused frozen author output-projection parameters
are excluded. Run `python inspect_parameters.py --method enzy` or `--method ses`
to count parameters independently.

## Configuration and training

Install the Anew dependencies and supply prepared data externally. The base
directory contains `assets/anewomni_official.ckpt`, `source/AnewOmni`, `inputs`,
`features` and `manifests/manifest.json`. Supply reference holdout data separately.
Dataset records contain input/feature paths and require a consistent accessible
layout. Runtime paths, scheduler account and logging identity are local settings.

```bash
python prepare.py --method ses --seed 1 --base /path/to/prepared-data \
  --holdout-root /path/to/reference-holdout --output runs/ses-seed1 --enable-training
export FUSION_ROOT="$PWD/runs/ses-seed1"
export FUSION_TRAIN_PYTHON=/path/to/environment/bin/python
bash "$FUSION_ROOT/train.sbatch"
```

Alternatively set `SLURM_ACCOUNT` and `GPU_PARTITION` for your scheduler and run
`python submit_training.py --root runs/ses-seed1`. W&B uses the logged-in workspace
or `WANDB_ENTITY`; preparation accepts `--wandb-entity` and `--wandb-project`.
The same entry point supports both methods and seeds1,123,42. Preparation never
submits jobs. Generated configs, logs and execution receipts remain untracked.

## Protocol and mapping

Only fusion parameters update. Native H/X/total loss, AdamW3e-4, effective batch32
through accumulation, gradient clipping1, warmup/wall-time cosine and a14,400-second
startup-inclusive A100 budget are retained. The fixed30k pool has10k logical
positions per modality and fixed antibody CDR identities. One shared checkpoint
is selected on the614-record validation set.253-target real-reference holdout loss
uses four fixed draws and never selects checkpoints. Disclose benchmark inspection
and known training-PDB overlap when reporting results.

Enzy retains six independent author adapters, attention output169, a169-to512
host bridge per layer, and the two-layer molecular projector with hidden1029,
SiLU and LayerNorm. SES uses one donor-query/native-KV update at the first native
normalized FFN input: hidden944, eight118-wide RoPE heads, dropout0, residual and
LayerNorm, with host dimension bridges. Native outer residual/vector operations
remain intact. Neither module adds a pair-feature branch or zero gate.

[Anonymous verification](VERIFICATION.json) covers native-checkpoint CPU forward/
backward, native loss, frozen states, replay and largest-record/effective-batch32
checks for all three seeds. It does not certify a completed training run, matched-
width CUDA verification or benchmark performance. [Full benchmark source](seed1_pipeline/README.md)
provides fixed253x30attempts, native scores and153-target protein-interface scores.
Each metric retains finite/missing denominators. Equal time and matched parameter
counts do not imply equal FLOPs or data presentations.

## Attribution

Pinned author sources and extraction hashes are recorded in `UPSTREAM_SOURCES.json`.
Consult upstream repositories for their original license and usage terms.

- [EnzyControl / EnzyAdapter](https://github.com/Vecteur-libre/EnzyControl/tree/2868406052a01a1a7a1c4ac636ca09f86a229394), [paper](https://arxiv.org/abs/2510.25132).
- [SES-Adapter](https://github.com/tyang816/SES-Adapter/tree/ec5bf619d665435c161493402495f89c95a2e756), [paper](https://doi.org/10.1021/acs.jcim.4c00689).
- [CAPLA](https://doi.org/10.1093/bioinformatics/btad049) is related biological cross-attention work; the existing single operator is not an exact CAPLA reproduction.
