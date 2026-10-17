# Single cross-attention vs multi-layer method

Training: one A100, 4 hours, same 30k manifest. Training seeds: baseline 42, historical method 20261007.
Actual GPUs: method NVIDIA A100-PCIE-40GB; baseline NVIDIA A100-PCIE-40GB. Same wall-clock budget does not establish matched FLOPs or throughput.
Evaluation: unchanged paired seeds; 253 targets x 30 attempted slots. Best values are bold, second values underlined; ties share rank.
Rank formatting does not establish statistical significance. Benchmark was previously inspected and has 19 known ligand training-PDB overlaps.

| Benchmark / metric | Method | Single cross-attention | Finite scores (method / baseline) |
|---|---:|---:|---:|
| lnr / structural/AAR | <u>6.37</u> | **6.49** | 2790 / 2790 |
| lnr / structural/complex_RMSD | **8.250** | <u>8.525</u> | 2790 / 2790 |
| rabd / structural/AAR | <u>15.78</u> | **16.05** | 1800 / 1800 |
| rabd / structural/complex_RMSD | **2.377** | <u>2.566</u> | 1800 / 1800 |
| crossdocked2020 / vina/dock | **-7.140** | <u>-7.116</u> | 1844 / 1829 |
| crossdocked2020 / chemical/qed | <u>0.572</u> | **0.580** | 1856 / 1846 |
| crossdocked2020 / success | **61.87** | <u>61.53</u> | 3000 / 3000 |
