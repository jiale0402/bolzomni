# Boltz Omni: joint folding and design PoC

Research code for a shared Omni checkpoint adapted from BoltzGen, with chemical identity feedback and joint folding/design training. This is an experimental source snapshot, not a packaged model release or evidence of superiority.

## Completed experiment

`experiments/native_condition_poc` contains the completed 400-update experiment: 128 training chains, 16 validation chains, a frozen trunk, and trainable diffusion, conditioning, feedback, and sequence-head parameters. Both tasks use the same final checkpoint and are compared with unchanged original BoltzGen weights.

| Metric | Original BoltzGen | Omni |
|---|---:|---:|
| Folding best-of-5 C-alpha lDDT (higher is better) | 0.59785 | 0.59713 |
| Folding best-of-5 TM-score (higher is better) | 0.59570 | 0.59184 |
| Design success | 18/24 | 19/24 |
| Design best-of-8 scRMSD (lower is better) | 0.7205 A | 1.0554 A |

All four paired 95% confidence intervals include zero. The small, previously inspected exploratory panel contains 8 folding targets and 6 design tasks. Design scores use ProteinMPNN sequence redesign followed by ESMFold; they do not validate Omni's raw sequence head. The original baseline was generated on A100, while repaired Omni generation used H100, an additional numerical limitation. **Improvement in both tasks was not demonstrated.**

Fixed-probe sequence cross-entropy decreased 11.96% on 4 training probes and 11.24% on 16 validation probes; the full objective decreased 6.08% and 5.23%. Structural validation loss reductions were only 0.283% (folding) and 0.049% (design). See [the report](experiments/native_condition_poc/RESULTS.md) and [machine-readable results](experiments/native_condition_poc/results/summary.json).

## Source map

- `experiments/native_condition_poc/`: native-conditioning model, training, generation, evaluation, audits, and completed report.
- `experiments/omni_joint/`: shared model components, data caching, metrics, W&B tracking, and historical experiment code.
- `experiments/official_poc/`: official-data preparation and supporting evaluator code; its reports describe an earlier run.
- `experiments/standard_bench/`: evaluator setup and compatibility helpers; its benchmark is distinct from the panel above.
- `experiments/noise_matched_poc/`: proposed continuation and raw-sequence evaluation code. Its presence does not establish execution or improvement; consult its status and plan.
- `scripts/`: shared loading, caching, generation, and pilot utilities.

## Dependencies and execution

The scripts were developed for Linux GPU nodes with Python 3.11 and compatible PyTorch/CUDA. Install the pinned upstream source at the expected path (upstream source is not redistributed here):

```bash
git clone https://github.com/HannesStark/boltzgen.git upstream
git -C upstream checkout a3149cf18eeb58648d1abbb27539bd73f746cdda
python -m pip install -e ./upstream
python -m pip install wandb numpy scipy matplotlib biopython biotite tmtools
export PYTHONPATH="$PWD/experiments/native_condition_poc:$PWD/upstream/src:$PWD/scripts:$PWD/experiments/omni_joint:$PWD/experiments/official_poc"
```

Obtain `boltzgen1_diverse.ckpt` and molecular assets separately under their upstream terms. `scripts/poc_model.py` records the original cluster's `ASSETS` path; adapt it to your installation. Dataset manifests contain original absolute paths, which must be mapped to your data layout. Data preparation starts with `experiments/official_poc/prepare_index.py` and `prepare_structures.py`; review their assumptions and manifests before use.

The design evaluator requires [MotifBench](https://github.com/blt2114/MotifBench) at `experiments/standard_bench/vendor/MotifBench`, revision `d5a714c6a376c701e29dfa66d42a1b13c5e339e0`, with Scaffold-Lab, ProteinMPNN, fair-esm 2.0.0 and OpenFold revision `4b41059694619831a7db195b7e0988fc4ff3a307`. See `experiments/standard_bench/setup.sbatch` and `setup_compat.py` for historical setup. Slurm scripts contain cluster-specific paths and environments and must be adapted before submission.

The completed workflow is `prepare.py`, `train.py`, `generate.py`, `refold.py`, and `report.py` in `experiments/native_condition_poc`. Preparation reuses audited original-baseline artifacts, which are not distributed here; a fresh installation must first generate and audit those inputs using supporting `official_poc` code. This is not a one-command reproduction bundle. Historical scripts and status notes are not current job status.

W&B tracking also persists local JSONL. Configure W&B through your environment or normal login, never source files. Offline runs remain pending sync.

## Release contents

Includes source, experiment manifests, and selected small result summaries. Excludes credentials, private machine configuration, downloaded datasets, model weights, generated structures, runtime environments, and raw tracking logs. Checkpoint hashes identify the evaluated artifact; checkpoint files are not included. No new model computation was performed for publication. Upstream projects retain their own licenses and asset terms.
