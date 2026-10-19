# EnzyAdapter benchmark pipeline

Generic source for native training, fixed-slot generation and scoring.
Configure the package in a separate run directory with external data paths,
logging identity and scheduler settings. Supply `reference_benchmark_config.local.json`
(schema: `reference_benchmark_config.example.json`) and `reference_training.local.json`
containing real reference-checkpoint training metadata. These files are untracked.
Set `FUSION_TRAIN_PYTHON`, `FUSION_SCORE_PYTHON`, `SLURM_ACCOUNT`, `GPU_PARTITION`
and `CPU_PARTITION` for the runtime. Set `training_enabled` in the local config
only when the run is ready; a `STOPPED.json` marker prevents progression.

`pipeline.py --advance --watch` advances this one run. Training preflight restores
disposable diagnostic updates and checks the production batch32 backward path.
Validation selects one shared checkpoint before benchmark generation. Inference
uses identical constructor dimensions and strict checkpoint loading.

Four disjoint workers cover31chunks:253targets x30fixed attempted slots, batch10,
100native diffusion steps. Native CPU scoring resumes verified caches; interface
scoring covers153protein targets and4590slots. Generation failures and each
metric's missing denominator are retained. Candidates are not resampled for quality.

Per-stage intent locks and an atomic progression mutex prevent duplicate jobs.
Interrupted started transactions require diagnosis. Every allocation records
source/data hashes, checkpoint identity, native loss, gradients and compute.
`FULL_COMPLETE.json` requires all stages, and `delivery/RESULTS.html` plus JSON
contain the reports. Server-side artifact verification and final scheduler
accounting are separate checks. Rosetta energies are computational REU, not
measured binding affinities.
