# Goal completion audit — 2026-09-28

User objective: first use a smaller training exposure to check for direct capability
destruction, then reach the previous experiment's scale; repair and retry if direct
destruction appears. The preceding goal turn made substantive progress: repaired source,
one completed training run, complete evaluation, and verified published artifacts.
This continuation re-read current source/configuration/results, queried Slurm again,
and rechecked W&B. It launched no training, inference, evaluation, or GPU job.

## Reference scale

"Previous experiment" is the immediately preceding failed official-data run,
`official_poc/runs/corrected-1729`, whose design result was 0/24. Its saved configuration
and the repaired run both specify:

| Dimension | Previous | Repaired |
|---|---:|---:|
| Official training chains | 128 | 128 |
| Validation chains | 16 | 16 |
| Optimizer updates | 400 | 400 |
| Effective batch / gradient accumulation | 8 / 8 | 8 / 8 |
| Diffusion learning rate | 5e-6 | 5e-6 |
| Added-module learning rate | 1e-4 | 1e-4 |
| Warmup updates | 40 | 40 |
| Trainable parameters | 280,139,956 | 280,139,956 |

The base-checkpoint and data-manifest SHA256 also match. Trunk remains frozen, with
full diffusion/conditioning/feedback/head updates and no LoRA. The corrected observation,
noise and alignment semantics intentionally differ. The CE coefficient is 0.189658 rather
than 3.337104, recomputed by the same initial-gradient calibration rule under the corrected
loss. These are not identical objectives, and no claim of identical parameter displacement
is made. The earlier 1000-/2000-update runs are not silently substituted as the scale target.

## Requirement-to-evidence check

| Requirement | Inspected evidence | Result and scope |
|---|---|---|
| Check actual training before reaching the previous full exposure | `runs/native-1729/validation-200.json`; source `train.py`; Quest log `logs/omni-native-flow-7716505.log` | After 200 updates, 2 validation proteins × folding/design each underwent the complete 500-step sampler. All 4 outputs were finite, with zero distance flags and maximum adjacent CA distances 3.8135–3.8366Å. Folding CA-lDDT was 0.9083/0.9688 versus initial 0.9119/0.9713. This is a mid-training damage check, not a design-success benchmark. |
| Continue to the previous experiment's full scale | The log records TRAIN 200, VALIDATION 200, TRAIN 210, TRAIN 400, VALIDATION 400 in that order; configurations above | Completed 400 updates at the previous data, batch, learning-rate and parameter scale. The 400-step run is already the full target exposure, not an unfinished smaller pilot. |
| Confirm useful final folding/design capability quantitatively | `results/summary.json`, all per-sequence CSVs, independent `completion_audit.json` | Original/repaired CA-lDDT 0.59785/0.59713, TM 0.59570/0.59184; design 18/24 vs 19/24. The catastrophic 0/24 design failure was not reproduced. No equivalence or universal non-regression claim is made. |
| Repair and retry if direct destruction occurs | `native_flow.py`, training/validation records, final full-generation and refold results | Prior relative-group hard projection and disabled-alignment path were replaced by native relative conditioning and the actual native sampler; replay uses all-coordinate noise. Direct catastrophic destruction did not recur during this repaired run, so another training retry is not indicated. |
| Added mechanism actually participates | `native_flow.py`, `results/mechanism_audit.json` | Sampler invokes feedback at each denoising step; token/pair projection norms changed from 0 to 0.726089/0.0818683, and identity-head weights changed. No fallback to the original model for final Omni evaluation. |
| Preserve experimental constraints and deliverables | Configuration, submissions, all audits and W&B artifacts | One shared final checkpoint, no control training/ablations; current user FairShare and eligibility checked for every submission. 80 folding predictions, 48 scaffolds and 384 refold sequences retained. Logs, curve, report and per-item scores delivered; W&B records finished and artifacts committed. |

Execution was one continuous training job with intermediate validation, not two separately
submitted experiments or a separately paused MPNN/ESMFold pilot. It must not be described
as having used that stricter procedure. The initial zero-update check alone is not the
evidence for preserving trained capability; the trained 200-update checks and final
400-update quantitative evaluation are the relevant evidence.

## Scientific conclusion and stopping point

This completes the requested progression to the previous full scale and its damage check.
It does **not** establish the paper's hoped-for joint improvement. All paired intervals
include zero; mean best-of-8 design scRMSD worsened from 0.7205Å to 1.0554Å. The reused
exploratory panel is small, folding uses the public validation list rather than the
unreleased Figure15 list, and original/new generation hardware was A100/H100. These limits
remain in the delivered report. Repeating training merely to obtain a positive outcome or
inventing a larger exposure would exceed this goal's specified scale.

Fresh Slurm query: jobs 7716505, 7716598, 7716617 and 7718381 are all COMPLETED, exit0:0.
Total allocated GPU time remains 0.6308333333 hours. The delivered report SHA256 remains
`d7812b4a236ee2bc69d76865a2e2e5ea0a874071849623a799876e817bcf0803`.

Fresh W&B API recheck in this continuation confirmed training state `finished`,
400 recorded updates, delivery state `finished`, and `audit/complete = 1`.
All four committed delivery-artifact file digests were verified again. No new
compute was submitted for this completion audit; the follow-up remains paused.
