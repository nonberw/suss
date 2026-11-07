# Pre-registration 2: results

Registered in `PREREGISTRATION_2.md` (tag `prereg2-94b6cf8`) before these analyses ran.

| Test | Estimate | 95% CI | p | p (Holm, 11 tests) | Supported |
|---|---|---|---|---|---|
| H6a|long60k: KV-only patching of the late blocks transfers tau >= 0.8 | 0.944 | [0.931, 0.958] | 9.999e-05 | 0.0008999 | yes |
| H6b|long60k: KV-only patching of block 1 transfers tau <= 0.3 | 0.133 | [0.102, 0.167] | 9.999e-05 | 0.0008999 | yes |
| H6c|long60k: fast-weight probe beats best residual readout by >= 0.15 R^2 (d_model dims) | 0.350 | [0.334, 0.368] | 9.999e-05 | 0.0008999 | yes |
| H6a|deep8: KV-only patching of the late blocks transfers tau >= 0.8 | 0.846 | [0.836, 0.855] | 9.999e-05 | 0.0008999 | yes |
| H6b|deep8: KV-only patching of block 1 transfers tau <= 0.3 | 0.010 | [0.003, 0.017] | 9.999e-05 | 0.0008999 | yes |
| H6c|deep8: fast-weight probe beats best residual readout by >= 0.15 R^2 (d_model dims) | 0.301 | [0.288, 0.314] | 9.999e-05 | 0.0008999 | yes |
| H6a|wide256: KV-only patching of the late blocks transfers tau >= 0.8 | 0.960 | [0.948, 0.971] | 9.999e-05 | 0.0008999 | yes |
| H6b|wide256: KV-only patching of block 1 transfers tau <= 0.3 | 0.100 | [0.063, 0.136] | 9.999e-05 | 0.0008999 | yes |
| H6c|wide256: fast-weight probe beats best residual readout by >= 0.15 R^2 (d_model dims) | 0.209 | [0.200, 0.219] | 9.999e-05 | 0.0008999 | yes |
| H7a: Jacobian closer to training-prior ridge than to new-prior Bayes | 0.001 (10/10 seeds) | [0.001, 0.002] | 6.731e-06 | 6.731e-05 | yes |
| H7b: Jacobian closer to training-prior ridge than to OLS | 0.009 (10/10 seeds) | [0.009, 0.009] | 1.13e-17 | 1.243e-16 | yes |

Verdicts (a hypothesis holds for the KV account only if it holds in every model): H6a supported, H6b supported, H6c supported, H7a supported, H7b supported
