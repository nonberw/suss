| Hypothesis | Estimate | 95% CI | p | p (Holm) | Supported |
|---|---|---|---|---|---|
| H1a: mean_t d(TF,ols) - d(TF,Bayes) > 0 | 0.2714 | [0.1809, 0.3785] | 9.999e-05 | 0.0009999 | yes |
| H1b: mean_t d(TF,gd1) - d(TF,Bayes) > 0 | 0.5517 | [0.4965, 0.6155] | 9.999e-05 | 0.0009999 | yes |
| H1c: mean_t excess_TF/excess_Bayes < 1.25 (t=4..47) | 1.59 | [1.564, 1.617] | 1 | 1 | no |
| H2a: max_l R2(theta_Bayes) > 0.8 | 0.1897 | [0.1743, 0.2123] | 1 | 1 | no |
| H2b: R2(theta_Bayes) - R2(theta_true) > 0 at l* | 0.004929 | [0.004245, 0.005615] | 9.999e-05 | 0.0009999 | yes |
| H3a: steering slope in [0.5,1.5] and r >= 0.7 | -3.441e-05 | [-9.02e-05, 2.488e-05] | 1 | 1 | no |
| H3b: subspace-patching transfer > 0.5 | 7.807e-05 | [-0.0001644, 0.0003356] | 1 | 1 | no |
| H4a: block: MSE(feature-RLS) - MSE(TF) > 0, t=24..47 | 0.01312 | [0.01279, 0.01345] | 9.999e-05 | 0.0009999 | yes |
| H4c: all identifiable targets R2 > 0.5 (intersection-union: max p) | nan | [nan, nan] | 1 | 1 | no |
| H4d: steering 1/m increases d v/d u on both systems (max p) | nan | [nan, nan] | 1 | 1 | no |
| H2c: argmax_l R2(moments) < argmax_l R2(theta_Bayes) in all seeds | l_mom=[2, 1, 2], l*=[4, 4, 4] | - | - | - | yes |
