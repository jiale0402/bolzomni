# Minimal Omni proof of concept

Status: authorized for execution, 2026-09-27. See RUN_STATUS.md for actual job state.
This file does not change or relabel the completed LoRA experiment.

## Question

Can one shared checkpoint using explicit identity prediction and feedback improve
both folding and design over the starting BoltzGen checkpoint?

## Two groups, one training run

User constraint: train Omni only. Do not add an original-architecture continued-
training control or any additional training ablation to this PoC.

| Group | Trainable components | Purpose |
| --- | --- | --- |
| A | None; original boltzgen1_diverse.ckpt | Starting performance |
| B | Full diffusion denoiser plus identity feedback and sequence head; trunk frozen | Omni treatment |

The single Omni training run starts from the original checkpoint, not the failed
adapted checkpoint. No LoRA. New modules in B have no counterpart in A. This
comparison measures the complete trained Omni model against the original model;
it does not isolate the effects of feedback, the identity objective, or additional
training.

Use the existing 1,613 eligible, filtered training examples. Use folding and motif
design prompts in a 1:1 mix. Fix the example/prompt stream, seed, effective batch
size and optimizer settings before training, and run 1,000 optimizer updates.
Record actual GPU time and memory.

Use native coordinate training and native-input teacher MSE on BOTH tasks in B.
B also learns masked identity prediction and consumes corrupted identity
states. Replay uses the teacher's native input with the same observed information,
never hidden sequence labels on a design prompt. Before updates, measure component
losses and shared-denoiser gradient scales on 32 fixed training batches. Freeze and
log the weight rule and resulting weights before training. No hyperparameter sweep.
Execution choices: the identity coefficient is the median coordinate/identity
gradient-norm ratio on the shared denoiser over 16 design calibration batches
(32 batches total, alternating tasks, chemical t=0.5). Both coordinate MSEs use
native sigma weighting and per-coordinate means; replay coefficient is 1 as a
declared equal-units tradeoff, not a tuned optimum. Replay gradients are initially
zero because teacher and student match, so an initial gradient ratio cannot set
that coefficient. Log component gradients after training updates as well.
Use AdamW at 1e-5 with 50-update linear warmup, weight decay 1e-4 and global norm
clip 1. Full diffusion includes the diffusion conditioning bridge; the static
trunk and input encoder remain frozen. Use BF16 autocast for training, preserving
the existing evaluation precision and sampling schedule.

The final step-1,000 Omni checkpoint is the only evaluated trained checkpoint.
Do not select a different checkpoint by task or by test outcome. Use the
existing validation split for separate folding/design monitoring; any failures
remain recorded rather than restarting silently with new settings.

## Fixed, small evaluation

- Folding: 20 chains from the existing 100-chain test manifest, selected by sorted
  SHA-256 of stable chain IDs before examining their outcomes. Primary metric:
  mean per-chain CA lDDT. Secondary: TM-score.
- Design: all 25 existing RFdiffusion motif-scaffolding tasks, four fixed candidates
  per task and group (100 candidates/group). Keep task coverage; reduce samples.
- Use the existing paired evaluator and exact original prompt, length, seed,
  500-step sampling, recycle, and AF2 settings. Reuse A's existing outputs only
  when all settings and source/checkpoint hashes match. Do not regenerate B
  selectively after seeing failures.
- Design uses each model's generated sequence. Existing structural pass criteria:
  AF2 design RMSD < 2 A, motif RMSD < 1 A, and pAE < 5. Retain validity checks and
  report Omni identity/geometry disagreement explicitly as raw failure. Show raw
  structural passes separately where evaluable; do not silently repair outputs.
- This is the existing adapted public assay, not an official leaderboard result.
  Test cases have been used previously: results are exploratory, not fresh
  independent confirmation. No claim of exclusion from base-model pretraining.

## Interpretation

Report B-minus-A on both tasks, with paired cluster bootstrap 95%
intervals (sequence clusters for folding, source PDB clusters for design). Report
all failure denominators and solved design tasks. Candidate count does not make
four samples from one design task independent targets.

- Both primary metrics improve over A: meets the direction of the user's practical
  target; intervals must exclude zero to establish improvement at this sample size.
- Positive estimates with intervals crossing zero: encouraging but inconclusive.
- Either task worsens or training fails: this PoC has not succeeded. A 1,000-update,
  one-seed failure does not establish that the general idea cannot work.

## Execution and records

### Repairs and checks within the single run

These are pending implementation requirements, not completed repairs. Full
denoiser training alone has not been shown to remedy the previous failure.

1. Fix the confirmed folding-only native_replay_eligible flag. Ensure design
   replay is evaluated on native prompts with matching information visibility.
   Check that it contributes gradients after student/teacher predictions diverge;
   zero MSE at identical initialization is expected and is not itself a failure.
2. Implement full-denoiser training with no LoRA. Verify frozen-trunk and trainable
   parameter manifests, and verify zero-feedback denoising parity with the original
   denoiser at identical inputs/noise. The previous two-example LoRA intervention
   implicated learned updates; it did not prove that full fine-tuning fixes geometry.
3. Within the same 1,000-update run, every 250 updates generate complete trajectories
   for four fixed folding and four fixed design validation prompts. Fix their IDs
   and seeds before training. Monitor folding CA lDDT, geometric UNKs, abnormal
   neighboring CA distances, returned identity/geometry agreement, and complete
   candidate validity. Log these alongside component loss/gradient scales. This
   is inference monitoring within the one run, not a second training experiment.
4. Keep the draft's progressive identity-reveal mechanism. The previous trained
   process failed to make final geometry match committed identities; that problem
   remains an empirical question. Do not substitute final head predictions or
   geometric identities for the actual generated identity state to hide failure.
5. Numerical failure aborts the run. If none of the four fixed design validation
   candidates at step 1,000 has valid geometry and endpoint identity agreement,
   report this run as unsuccessful and skip the costly full AF2 test. Preserve the
   checkpoint and evidence; do not select an earlier checkpoint or restart silently.
   This small validation gate is a compute stop rule, not proof against the idea.

Quest via Slurm is preferred. Full-denoiser throughput and memory must be measured;
do not extrapolate a completion time from the earlier LoRA run. Run one training
seed, no extra module ablations or task families. Log config, data/source hashes,
raw and weighted losses, calibration gradient scales, separate folding/design
validation metrics, fixed checkpoint rule, checkpoint artifacts, GPU usage and
final results to W&B and local JSONL. If offline, explicitly mark pending sync.
Never log credentials. This proposal is not a report of completed execution.
