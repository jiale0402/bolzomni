# Small paired standard benchmark

Evaluate the frozen original BoltzGen and the single final full-diffusion Omni
checkpoint. No additional training or checkpoint selection.

| Task | Public subset | Metric |
|---|---|---|
| Folding | CASP16, 10 targets: 3 short / 4 medium / 3 long; unique target families | Biotite Cα lDDT; TM-align TM-score against resolved experimental structure |
| Motif scaffold design | MotifBench, 6 cases: 2 per motif-segment-count group, 2 candidates each | Official ProteinMPNN 8 sequences + ESMFold; scaffold succeeds if one sequence has both scRMSD < 2 Å and motif RMSD < 1 Å |

Selection is deterministic, before predictions. CASP targets require at least
70% resolved Cα and no adaptation train/validation homology hit at 30% identity,
80% query coverage. Motif tasks span the shortest and longest distinct PDBs in
each segment-count group. This is a stratified screening subset, not a random
estimate of the entire benchmark or a full leaderboard submission. Pretraining
exposure of the base model has not been excluded.

Both arms use identical prompts, paired generation seeds, 500 diffusion steps,
3 recycles and the original model's sampling scales. Folding is sequence only.
The reference structure never enters folding model inputs. Design prompts give
only the specified motif, with the same deterministic gap placement in both arms.

Design sequences are redesigned by the official evaluator. This measures scaffold
designability, not the quality of Omni's own sequence head. Own-sequence/geometry
agreement is preserved as a diagnostic and does not gate the standard score.

Report both model scores and paired deltas, plus target-level bootstrap intervals
(10 folding targets; 6 design tasks). Two candidates and eight evaluation sequences
are not independent tasks. Technical scoring failures must be resolved and not
silently discarded. Both independent tasks must improve to support the intended
claim. A small subset can reject a large degradation or remain inconclusive; it
cannot establish universal superiority.

Evaluation sources are pinned: MotifBench d5a714c6a376c701e29dfa66d42a1b13c5e339e0;
OpenFold 4b41059694619831a7db195b7e0988fc4ff3a307 (ESM's documented version);
fair-esm 2.0.0. Compatibility changes disable unused PyMOL imports, make Python
3.11 dataclass defaults legal, and defer the unused optional CUDA extension import.
The public ProteinMPNN CLI seed is fixed to 2 in both arms. Metric formulas and
evaluator weights remain unchanged.

Quest jobs: setup.sbatch (CPU dependencies), generate.sbatch (GPU paired predictions),
refold.sbatch (GPU official design scoring and report). Track W&B and local JSONL.

Sources: https://predictioncenter.org/download_area/CASP16/
and https://github.com/blt2114/MotifBench
