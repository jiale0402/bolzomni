# Single cross-attention baseline (seed 1)

This directory contains the exact training/evaluation scripts submitted for the
parameter-matched single-fusion baseline. There is one conditioning update
`c' = c + cross_attention(c, PairFormer single, PairFormer pair)`, followed by
the unchanged native AnewOmni epsilon network. The Anew backbone is frozen.
Only the existing projection widths are increased, with no added fusion depth.

- Baseline: 4,028,016 trainable parameters; multi-layer method: 4,027,712.
- Training seed: **1**. Historical method training seed: **20261007**.
- One A100, **14,400 seconds (4 GPU-hours)** from job startup, including
  startup/preflight, training, validation and training-time holdout reporting.
  The Slurm allocation has extra bookkeeping time; the trainer enforces the
  four-hour experiment budget. Actual elapsed times are reported independently.
- Same fixed 30,000 logical training records (10k each modality), same PairFormer
  features, same frozen antibody CDR identities; no feature extraction resumed.
- Same AdamW/LR/batch/gradient clipping/warmup/cosine recipe as the method.
- One shared checkpoint, selected on the unchanged **614-record validation**.
- Native H/X/total validation and real-reference benchmark holdout losses are
  reported per modality at initialization, every 500 steps and finally.
- Generation: **253 targets x 30 attempted slots**, batch 10, gamma 1, native
  100-step diffusion and original decoding/scorers. No quality-based retries.
- Keep the original **evaluation seed 20260814** and paired per-slot hashes;
  validation and holdout use their original paired seeds too. Seed 1 changes
  training, not the fixed evaluation protocol.
- Independent W&B runs and local JSONL, with configuration/source/data hashes,
  actual optimizer loss, gradient norm, checkpoint artifacts and compute usage.
  Offline W&B is retained and identified as pending sync when necessary.

The benchmark has been inspected previously and has 19 known ligand training-PDB
overlaps. The comparison discloses those and the different training seeds.
Matching parameter count and wall-clock hardware budget does not match FLOPs,
optimizer-step counts, or data presentations; actual counts are retained.

## Files and execution

`code/fusion.py` is the baseline. `code/train.py` is the unchanged native
multi-layer-method training harness with this module substituted.
`code/config.json` contains seed 1 and the fixed comparison settings.
`runner/` contains the pinned real benchmark implementation/scorers, with the
single-fusion module substituted and donor projection caching disabled.
`SOURCES.json` pins all executed script hashes.

The scripts use the existing verified Quest assets/data/environments, which are
not included in this code-only upload. Set `FUSION_ROOT` to the staged run root.
The active run is `/projects/p53127/anewomni_work/single_cross_attention_20261015/seed1`.

```bash
python3 "$FUSION_ROOT/pipeline.py" --advance
```

The idempotent pipeline submits training, then four disjoint generation workers,
then CPU scoring and a two-model comparison. Each Quest submission freshly
checks the `jck3114` association's FairShare for **both p53127 and p53252**, checks
Slurm resource eligibility, and uses the eligible higher-FairShare account.
Existing submission receipts prevent duplicate work. Do not delete receipts or
restart training to obtain a different result.

Final files: `training/selection.json`, `training/COMPLETE.json`,
`benchmark/REPORT.json`, `comparison/RESULTS.json`, `comparison/RESULTS.md`.
They include checkpoint identities, metric denominators/missing scores,
modality H/X/total losses, actual compute, and paired target-bootstrap intervals.
Best displayed metric values are bold and second-best underlined; ties share
rank. Rank formatting does not imply statistical significance.

The earlier readiness check passed on real three-modality data and a frozen
original checkpoint on an idle Orion A6000; peak allocated memory was 3.13 GiB.
That short diagnostic is not a training result. Formal training uses the A100
budget above and repeats the native correctness checks at its own seed.

[Readiness W&B run](https://wandb.ai/jialechen2025-northwestern-university/anewomni-pairformer-fusion/runs/single-cross-attention-matched4m-20261015-native-smoke)

[Seed-1 training W&B run](https://wandb.ai/jialechen2025-northwestern-university/anewomni-pairformer-fusion/runs/single-cross-attention-matched4m-seed1-20261015-training-native)

The training W&B run is created when the queued GPU job actually starts; a Slurm
submission is not evidence that training has begun.
