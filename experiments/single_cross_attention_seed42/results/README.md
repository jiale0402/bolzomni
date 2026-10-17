# Completed seed-42 repeat

Training, the full fixed benchmark, comparison and interface scoring are complete.
Both baseline seeds are retained in MAIN_TABLE_BOTH_SEEDS.json. No best-seed selection.

Seed 42: one A100-PCIE-40GB, 14,223.7749 startup-inclusive seconds within the
14,400-second budget; 3,426 optimizer updates / 109,584 presentations. Historical
method training seed: 20261007. Seed 1 used A100-SXM4-80GB; seed 42 and the method
used A100-PCIE-40GB. Matched wall-clock time does not match FLOPs or presentations.

Generation: 253 targets, 7,590 fixed attempts, 6,436 exports and 1,154 native
validity/filter failures. All 253 scoring receipts, all slots and 6,436 candidate
score-file hashes were verified. All four generation workers and the CPU scoring
job completed. W&B server verification confirms 35 finished runs with committed
artifacts, including training, 31 chunks, native scoring, comparison and interfaces.

RAbD DockQ/iRMSD/contact: 59 targets / 1,770 finite candidates; 30 scores are
undefined because reference rabd_055_3uzq_B_A_a has no native heavy-atom contact.
Backbone interface lDDT covers all 60 targets / 1,800 slots. Ligand finite native
Vina scores: 1,829 / 3,000; chemical scores: 1,846 / 3,000. Missing values retained.

STATISTICS_BOTH_SEEDS.json covers all 54 metric-by-seed comparisons with Holm
correction, 50,000 target bootstraps and 200,000 target-group swaps. Antibody
complex RMSD, DockQ, interface RMSD and backbone interface lDDT improve against
both baseline seeds. This does not measure training-seed variability of the method.
The benchmark was previously inspected and has 19 known ligand training-PDB overlaps.

Exact analysis source versions are preserved in ../analysis/. Reproduction reads
the original hash-pinned local/Quest score arrays; those arrays are preserved in
the experiment evidence, rather than regenerating candidates. The executed
training/generation/scoring source ledger remains SOURCES.json at commit f36652b.
Post-hoc result files and analysis scripts are outside that execution ledger.

[Official main table](https://jiale0402.github.io/all-atom-omni/#trimodal-results)
