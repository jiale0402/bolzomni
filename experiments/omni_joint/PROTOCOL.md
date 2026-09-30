# Shared Omni adaptation: design and folding

Status: implementation and baseline qualification; no new training or performance claim.
This experiment supersedes the local-completion pilot in POC_PLAN.md. Old results remain separate.

## Authorized question
Does ONE jointly trained checkpoint outperform its original BoltzGen checkpoint in BOTH
folding and design? No task-specific checkpoint, head selection by task, or task-specific
adapter is allowed. The matched no-feedback ablation is deferred; improvement cannot be
attributed uniquely to late feedback. Protein assays do not establish other output classes.

## Architecture contract
- Retain the pinned BoltzGen trunk and original denoiser weights, frozen.
- Compile independent chemical/coordinate observation masks before static encoding.
- Cache observed context only. Feed chemical state and both corruption levels into shared
  token and low-rank pair updates. Apply pair updates before the frozen pair conditioner,
  including their atom-encoder and token-attention paths.
- Static contexts may be cached lazily at first training access. This does not change the
  locked dataset, prompt masks or training objective; every new entry checks hidden-input invariance.
- Add rank-8 zero-output LoRA updates to denoiser token-attention query/value projections.
- Read chemical logits from non-detached denoiser features. Reveal sampled identities
  progressively and feed them into subsequent score evaluations.
- Preserve hard observations after each coordinate/chemical update in their declared frame.
  Native soft distance groups remain distinguished from absolute observations.
- Compare endpoint categorical identity with atom14 geometric decoding. Disagreement is a
  raw failure, never silently repaired or replaced by inverse folding.
- Train completion coordinate loss, masked identity cross-entropy and native-input replay.
  Ordinary protein bonds are declared by layout; no claim of learned ligand graph generation.

## Data and evaluation contract
- Native source: a3149cf18eeb58648d1abbb27539bd73f746cdda; checkpoint boltzgen1_diverse.ckpt.
- Training: 1,622 eligible canonical complete CATH training examples before homology filtering
  and 200 validation examples. The original target of 2,000 exceeded the eligible pool;
  on 2026-09-27 the count was reduced without relaxing quality criteria or selecting by performance.
  Mix sequence-observed folding, motif-conditioned design, inverse folding and mixed masks.
  Actual eligible counts and exclusions must be recorded before training.
- Folding: target 100 eligible CATH topology-test chains, excluding previously inspected pilot
  test chains. Fixed sequence-only prompts (no native geometry, MSA or template); same inputs
  for both models. This is a public-data evaluation, not a claimed official folding benchmark.
- Design: RFdiffusion Supplementary Table 9, all 25 motif-scaffolding tasks. Pin exact task
  files and evaluator revision before generating the test set. Do not silently omit hard tasks.
- Remove exact and near-sequence overlaps between adaptation data and evaluation; record
  clustering settings and unresolved base-pretraining exposure. Never call these unseen to
  the base model without evidence.
- Folding primary: mean per-chain CA lDDT (15 A neighborhood; 0.5,1,2,4 A thresholds), with
  independently implemented reference cross-check. Secondary TM-score, aligned CA RMSD.
- Design primary: mean per-task raw candidate success, with AF2 design RMSD <2 A,
  native motif RMSD <1 A, and pAE <5, using exact published atom/alignment conventions.
  Additional constraints: valid identities and identity/geometry agreement for Omni.
  Report raw structural passes separately so consistency failures remain visible.
- Own generated sequence for each model; no external inverse-folding replacement.
  This adapts the public structural assay to sequence+structure generation and is not a
  reproduction of a backbone-only pipeline leaderboard.
- Initially 16 design candidates/task/model and 1 fixed-seed folding sample/chain/model;
  identical sampling evaluations, recycles, size conditions and selection budgets.
  One training seed is a pilot limitation, not evidence of seed robustness.
- Report paired differences and 95% target/chain bootstrap intervals (10,000 replicates),
  solved tasks, unique successful sequences, invalid outputs and all failure denominators.
  Design length variants from the same PDB are one resampling cluster (25 tasks, 17 PDBs),
  not independent targets. Folding resamples sequence clusters after the overlap audit.
- Success requires BOTH primary differences >0 with BOTH interval lower bounds >0.
  Loss reduction, one-task improvement, or intervals crossing zero do not meet the objective.
- Select ONE checkpoint using a preregistered joint validation criterion. Test metrics cannot
  select checkpoints, weights, schedules, evaluators, thresholds or task exclusions.

## Execution gates
1. Diagnose prior baseline failure against full native inference, without adapter training.
2. Test leakage, zero-output reproduction, gradients, pair-path influence, chemical feedback,
   observed-field preservation and endpoint consistency. Preserve failed audit records.
3. Verify independent evaluators and qualify baseline on validation examples; profile cost.
4. Freeze manifests/configuration/hashes and validation selection rule, then train.
5. Run both held-out assays with the SAME selected checkpoint and produce paired tables.

Initial GPU audit wall-time cap: 1 GPU-hour. First-run total ceiling: 24 GPU-hours, including
cache, training, teacher, generation and evaluation. Measure throughput before committing;
if insufficient, report the shortfall instead of silently weakening the experiment.

## Required experiment tracking
Weights & Biases project: boltz-omni-joint (default authenticated workspace unless configured).
Every training run records source/data hashes, architecture/loss/sampling settings, losses,
separate folding/design validation scores, common checkpoint selection, checkpoint artifacts,
GPU type/hours and peak memory. Local JSONL mirrors scalar records. Offline fallback must be
explicitly labeled pending sync. Test runs are separate from training/validation selection.
Credentials are read only through wandb's normal credential mechanism and never logged.

Reference: https://doi.org/10.1038/s41586-023-06415-8
