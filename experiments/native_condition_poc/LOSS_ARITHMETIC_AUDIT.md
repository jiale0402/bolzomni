# Loss arithmetic audit

Saved scalar logs only. No model execution.

`total = (folding + design) / 2`; identity CE weight = 0.1896582197810322.

| Check | Maximum absolute residual |
|---|---:|
| training_total | 0 |
| training_components | 6.04823071626e-08 |
| weighted_identity | 3.24561633125e-08 |
| rolling_mean_total | 1.11022302463e-16 |
| fixed_probe_components | 0 |
| prior_plotted_total | 0 |
| prior_plotted_components | 0 |
| prior_plotted_structural_components | 0 |

All 400 training updates and all 381 complete 20-update moving windows passed.
Fixed probes were checked against every saved per-sample component at 0/100/200/300/400.

| Split | Update | Folding full loss | Design full loss | Total |
|---|---:|---:|---:|---:|
| train | 0 | 0.274605203 | 0.969252639 | 0.621928921 |
| validation | 0 | 0.306975310 | 0.996618414 | 0.651796862 |
| train | 100 | 0.274344741 | 0.935798928 | 0.605071835 |
| validation | 100 | 0.306508660 | 0.963092352 | 0.634800506 |
| train | 200 | 0.274292067 | 0.915918165 | 0.595105116 |
| validation | 200 | 0.306430404 | 0.944820622 | 0.625625513 |
| train | 300 | 0.274387626 | 0.904320832 | 0.589354229 |
| validation | 300 | 0.306698574 | 0.935432124 | 0.621065349 |
| train | 400 | 0.273899681 | 0.894345166 | 0.584122423 |
| validation | 400 | 0.306318630 | 0.929039601 | 0.617679115 |

The fixed-probe full totals decrease from update 0 to 400 in both splits.
The earlier multi-panel figures mixed three measurement protocols visually: changing-batch training, fixed-probe full objective, and structural-only objective. Their arithmetic must be compared within the same protocol.
No arithmetic discrepancy beyond floating-point rounding was found. Gradient conflict cannot explain a violation of the scalar weighted-sum identity.

The checked_*_loss.png figures each use only one fixed split and the complete weighted objective. Native structural references remain in train_validation_native_baselines.png; no native CE or native full joint scalar is invented.
