# SUSS trained models

All checkpoints used in the paper. Load with `torch.load(path, map_location='cpu')`; GPT checkpoints hold
`cfg` (a `suss.model.GPTConfig` as a dict) and `state` (the state dict). See `notebooks/quickstart.ipynb`.

| file | what | parameters | MB |
|---|---|---|---|
| `runs/a/seed0.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed1.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed2.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed3.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed4.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed5.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed6.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed7.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed8.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/a/seed9.pt` | A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication | 800,258 | 3.2 |
| `runs/b/block_tf_seed0.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,258 | 3.2 |
| `runs/b/block_tf_seed1.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,258 | 3.2 |
| `runs/b/block_tf_seed2.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,258 | 3.2 |
| `runs/b/bouncer_tf_seed0.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,258 | 3.2 |
| `runs/b/bouncer_tf_seed1.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,258 | 3.2 |
| `runs/b/bouncer_tf_seed2.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,258 | 3.2 |
| `runs/b/pendulum_tf_seed0.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,515 | 3.2 |
| `runs/b/pendulum_tf_seed1.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,515 | 3.2 |
| `runs/b/pendulum_tf_seed2.pt` | B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats) | 800,515 | 3.2 |
| `runs/b/block_mlp_seed0.pt` | B: no-context MLP baseline | 67,330 | 0.3 |
| `runs/b/block_mlp_seed1.pt` | B: no-context MLP baseline | 67,330 | 0.3 |
| `runs/b/block_mlp_seed2.pt` | B: no-context MLP baseline | 67,330 | 0.3 |
| `runs/b/bouncer_mlp_seed0.pt` | B: no-context MLP baseline | 67,330 | 0.3 |
| `runs/b/bouncer_mlp_seed1.pt` | B: no-context MLP baseline | 67,330 | 0.3 |
| `runs/b/bouncer_mlp_seed2.pt` | B: no-context MLP baseline | 67,330 | 0.3 |
| `runs/b/pendulum_mlp_seed0.pt` | B: no-context MLP baseline | 67,843 | 0.3 |
| `runs/b/pendulum_mlp_seed1.pt` | B: no-context MLP baseline | 67,843 | 0.3 |
| `runs/b/pendulum_mlp_seed2.pt` | B: no-context MLP baseline | 67,843 | 0.3 |
| `runs/c/bc_seed0.pt` | C: controller after the DAgger warm start (imitation only) | 800,129 | 3.2 |
| `runs/c/bc_seed1.pt` | C: controller after the DAgger warm start (imitation only) | 800,129 | 3.2 |
| `runs/c/bc_seed2.pt` | C: controller after the DAgger warm start (imitation only) | 800,129 | 3.2 |
| `runs/c/seed0.pt` | C: final controller (warm start + log-cost fine-tuning, validation-selected checkpoint) | 800,129 | 3.2 |
| `runs/c/seed1.pt` | C: final controller (warm start + log-cost fine-tuning, validation-selected checkpoint) | 800,129 | 3.2 |
| `runs/c/seed2.pt` | C: final controller (warm start + log-cost fine-tuning, validation-selected checkpoint) | 800,129 | 3.2 |
| `runs/c/seed0_last.pt` | C: last fine-tuning checkpoint (not selected) | 800,129 | 3.2 |
| `runs/c/seed1_last.pt` | C: last fine-tuning checkpoint (not selected) | 800,129 | 3.2 |
| `runs/c/seed2_last.pt` | C: last fine-tuning checkpoint (not selected) | 800,129 | 3.2 |
| `runs/c/seed0_raw.pt` | C: pilot, warm start + raw cost (diverged; selected = warm start) | 800,129 | 3.2 |
| `runs/c/seed0_raw_last.pt` | C: pilot, warm start + raw cost (diverged; selected = warm start) | 800,129 | 3.2 |
| `runs/c/seed0_cvar.pt` | C: pilot, warm start + CVaR (diverged; selected = warm start) | 800,129 | 3.2 |
| `runs/c/seed0_cvar_last.pt` | C: pilot, warm start + CVaR (diverged; selected = warm start) | 800,129 | 3.2 |
| `runs/c/fixed_seed0.pt` | C: best constant-gain baseline (a 1x2 gain matrix) | 2 | 0.0 |
| `runs/c/fixed_seed1.pt` | C: best constant-gain baseline (a 1x2 gain matrix) | 2 | 0.0 |
| `runs/c/fixed_seed2.pt` | C: best constant-gain baseline (a 1x2 gain matrix) | 2 | 0.0 |
| `runs/scale/deep8.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 1,593,346 | 6.4 |
| `runs/scale/long60k.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 800,258 | 3.2 |
| `runs/scale/n4m2_s0.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 800,900 | 3.2 |
| `runs/scale/n4m2_s1.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 800,900 | 3.2 |
| `runs/scale/n4m2_s2.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 800,900 | 3.2 |
| `runs/scale/small64.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 103,554 | 0.4 |
| `runs/scale/wide256.pt` | Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2} | 3,173,378 | 12.7 |
