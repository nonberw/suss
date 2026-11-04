# SUSS pre-registration

Frozen at the commit that adds this file (2026-11-05), **before any main experiment was run**. Pilot runs are allowed only to check that code runs and that training converges. Pilots use seed 99, which is never reported. Any change after this commit goes in `docs/DEVIATIONS.md` with a reason.

Research question (sharpened, see `docs/LITERATURE.md`): when a transformer adapts to unknown dynamics in-context, does it (i) accumulate the estimator's sufficient statistics and then solve for a Bayes-optimal estimate, and (ii) when trained as a controller, inject the excitation that closed-loop identifiability theory says is necessary? Are the internal quantities causal?

## Shared model and training

- GPT-style causal decoder, pre-LayerNorm, d_model = 128, 4 layers, 4 heads, MLP width 512, GELU, learned positional embeddings, no dropout. A linear token embedding and a linear output head.
- AdamW (lr 3e-4, weight decay 0.01, betas 0.9/0.95), 500 warm-up steps, cosine decay to 10%, grad-clip 1.0, batch 256.
- Training seeds **{0, 1, 2}** for every model. Test sets use fixed seeds that are disjoint from training: 10_000 + experiment index.
- The **residual stream** at layer ℓ ∈ {0,…,4} means the input to block ℓ+1 (ℓ = 0: embeddings, ℓ = 4: final, before the last LayerNorm).

## Experiment A: in-context prediction of linear systems (exact optimum known)

- System: x_{t+1} = A x_t + B u_t + w_t, x ∈ R², u ∈ R, w ~ N(0, σ² I), σ = 0.1.
- Prior: A_ij ~ N(0, 0.4²), B_i ~ N(0, 0.5²) i.i.d., rejected unless spectral radius ρ(A) < 0.95. Inputs u_t ~ N(0,1) i.i.d. x_0 ~ N(0, I).
- Sequence: T = 48 tokens, token t = (x_t, u_t), target x_{t+1}. Loss: MSE summed over both coordinates, averaged over all positions. 20 000 steps; each batch is freshly sampled.
- Test: 4096 episodes.
- References, computed per position t from the t+1 pairs (z_s, x_{s+1}), s ≤ t−1, plus the current query z_t:
  - **Bayes**: exact posterior mean under the truncated prior. Monte Carlo with 2048 samples from the untruncated Gaussian posterior, keeping those with ρ(A) < 0.95 (fall back to Ridge if none accepted).
  - **Ridge**: posterior mean under the untruncated Gaussian prior (= RLS with prior initialisation).
  - **OLS**: minimum-norm least squares.
  - **GD-1**: one gradient step from θ = 0 on the context squared loss, step size η/(t+1) with η tuned on 512 separate episodes.
  - **Zero**: predict 0.
- Metrics: MSE(t); excess(t) = MSE(t) − nσ² (nσ² = 0.02); algorithmic distance d(M, E)(t) = E‖ŷ_M − ŷ_E‖².

**H1a (confirmatory).** Averaged over t = 1…47, d(TF, Bayes) < d(TF, OLS).
**H1b (confirmatory).** Averaged over t = 1…47, d(TF, Bayes) < d(TF, GD-1).
**H1c (confirmatory).** The mean over t = 4…47 of excess_TF(t)/excess_Bayes(t) is < 1.25 (upper 95% CI bound < 1.25).
**H1d (exploratory).** t·excess_Bayes(t) approaches n·p·σ² = 0.06 (p = 3 regressors) at large t, as Theorem 1 predicts. The transformer's excess follows the same law.

## Experiment D on A: mechanism

Probes: ridge regression from the residual stream (one layer at a time) at positions t = 4…47, pooled across positions. The ridge α comes from 5-fold CV on {1e-3, 1e-2, …, 1e2}. Probes are trained on 2048 test episodes and scored on the other 2048 (held-out R², averaged over target coordinates). Targets:
- θ_true (6 numbers), θ̂_Bayes(t) (6), θ̂_OLS(t) (6);
- sufficient statistics: S_t/(t+1) (upper triangle of Σ z zᵀ, 6 numbers) and C_t/(t+1) (Σ z x_{next}ᵀ, 6 numbers);
- posterior uncertainty: log tr P_t.

**H2a (confirmatory).** max_ℓ R²(θ̂_Bayes) ≥ 0.8 (lower 95% CI bound ≥ 0.8).
**H2b (confirmatory).** At the layer ℓ* that maximises R²(θ̂_Bayes), R²(θ̂_Bayes) > R²(θ_true). The model encodes its estimate, not the truth.
**H2c (confirmatory, deterministic criterion).** In each of the 3 seeds, argmax_ℓ R²(moments) < argmax_ℓ R²(θ̂_Bayes): statistics peak strictly earlier than the estimate. Supported only if this holds in all three seeds.

Interventions (at the last position t = 47 unless stated):
- **Steering.** Let W be the 6×128 θ̂-probe at ℓ*. Add δ·W⁺e_k to the last-token residual for coordinate k, with δ = ±0.25. Bayes-predicted change in the output: Δŷ_i = δ·z_j when k indexes A_ij or B_i (z_j is the matching query entry). Regress the observed Δŷ on the predicted Δŷ, pooled over episodes, k and sign.
  **H3a (confirmatory).** The slope's 95% CI lies inside [0.5, 1.5], and Pearson r ≥ 0.7.
- **Subspace patching.** Pairs (A, B) of test episodes. A′ = A's first 47 tokens followed by B's last token. Replace the projection of B's last-token residual at ℓ* onto the row space of W with that of A′. Transfer τ = ⟨ŷ_patch − ŷ_B, ŷ_A′ − ŷ_B⟩ / ‖ŷ_A′ − ŷ_B‖².
  **H3b (confirmatory).** The mean τ_sub is > 0.5 (lower 95% CI bound > 0.5).
- **Exploratory.** Layer-wise patching of all context positions (τ_ctx(ℓ)) and of the last token (τ_last(ℓ)); per-head patching; per-head mean-ablation loss; attention entropy and previous-token score per head; R² vs t.

## Experiment B: nonlinear and contact systems in MuJoCo (optimum unknown)

Two systems. Control step dt = 0.05 s (physics 0.01 s × 5). Process noise is an unobserved random generalized force, N(0, s_p²) per physics step. T = 48, 60 000 training episodes and 4096 test episodes per system, generated once from fixed seeds.
- **P, pendulum:** point mass on a massless 0.5 m rod. m ~ U[0.5, 2], joint damping c ~ U[0.02, 0.3], gravity g ~ U[4, 16]. Torque u ~ N(0, 2²) clipped to ±5. s_p = 0.2. Observation (sin q, cos q, q̇). q_0 ~ U[−π, π], q̇_0 ~ N(0,1).
- **K, pushed block:** a box on a plane, slide joint along x, Coulomb contact friction μ ~ U[0.1, 1.0], m ~ U[0.5, 3]. Force F ~ U[−15, 15] N i.i.d. per step. s_p = 0.2. Observation (x, ẋ). This gives stick–slip.
- Token t = (obs_t, u_t), target obs_{t+1}. 20 000 steps per seed over the fixed dataset.
- Baselines: **oracle-sim** (MuJoCo with the true parameters and no noise, stepped from the true state); **nominal-sim** (mid-range parameters); **feature-RLS** (ridge on physics features: P regresses Δq̇ on [u, q̇, sin q, 1]; K regresses Δẋ on [F, sign ẋ, ẋ, 1]; positions are integrated with the trapezoid rule); **no-context MLP** (sees only the current token).
- Metric: next-observation MSE with each dimension normalised by its test variance.

**H4a (confirmatory).** On K, the transformer's MSE averaged over t = 24…47 is < feature-RLS.
**H4b (exploratory).** On P, where feature-RLS has the true model class, we expect TF ≥ feature-RLS.
**H4c (confirmatory).** The probe R² at t = 47 (best layer, held-out) for the identifiable combinations is ≥ 0.5 (lower CI): P: g and 1/m; K: μ.
**H4d (confirmatory).** Steering along the 1/m probe direction (+1 target SD) increases ∂ŷ_{velocity}/∂u (autograd with respect to the last token's input) on both systems. The mean change is > 0 with the 95% CI excluding 0. We also report the fraction of episodes with the correct sign.

## Experiment C: in-context control

**C1, linear adaptive regulation (main).**
- Plant as in A, but A = 0.9 I + N(0, 0.25²) entrywise, truncated to ρ(A) < 1.1 (some plants are open-loop unstable). B = r(cos φ, sin φ), r ~ U[0.3, 1], φ ~ U[0, 2π): the direction and sign of the input are unknown. σ = 0.1, x_0 ~ N(0, I), horizon H = 48.
- Cost J = Σ_{t<H} (‖x_t‖² + 0.1 u_t²) + ‖x_H‖². Every controller's actions are clipped to [−5, 5].
- Transformer controller: token t = (x_t, u_{t−1}), output u_t = 5 tanh(·). Trained by back-propagation through the differentiable plant on J, fresh plants per batch, 6000 steps, 3 seeds.
- Baselines:
  - **oracle** (finite-horizon LQR with the true A, B);
  - **fixed** (the best constant gain K over the prior, optimised by gradient on training plants);
  - **CE** (certainty equivalence: ridge estimate each step, then finite-horizon LQR on the estimate, no dither);
  - **CE+dither** (Gaussian dither s_t = s_0/√(t+1); s_0 and the ridge λ tuned on 1024 training plants over s_0 ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6} and λ ∈ {0.01, 0.1, 1});
  - **SysID-then-control** (explore with u ~ N(0, s_e²) for the first k steps, then CE without dither; k ∈ {2, 4, 8}, s_e ∈ {0.5, 1, 2} tuned).
- Test: 4096 plants, with noise shared across controllers (paired).
- Metric: normalised cost J/J_oracle. We report the mean, the median and the failure rate (J/J_oracle > 10).

**H5a (confirmatory).** Mean normalised cost: TF < fixed.
**H5b (confirmatory).** TF < CE (no dither).
**H5c (exploratory, two-sided).** TF vs CE+dither and vs SysID-then-control.
**H5d (confirmatory, the dual effect).** The information gained in the first 8 steps, log det(Λ₀ + σ⁻² S_8), is larger under TF than under CE (paired by plant).
**H5e (exploratory).** Implicit system identification: probe R² of the controller's residual stream for B, and for the ridge estimate B̂_t computed from its own history. Steering along the B̂ probe changes u_t in the direction LQR would predict.

**C2, MuJoCo block positioning (secondary; it may be dropped for compute, and this is documented).** Bring block K from rest to a random target in [−1, 1] m with unknown m and μ. Controllers:
- **oracle** (PD plus exact friction compensation with the true m, μ);
- **fixed** (nominal parameters);
- **SysID+control** (feature-RLS estimates plugged into the same law);
- **TF** (behaviour cloning plus 3 DAgger rounds on oracle actions, conditioned on history).
- **H6 (secondary).** TF < fixed in final position error; TF vs SysID+control two-sided.

## Statistics

- All CIs are 95% percentile CIs from a **hierarchical bootstrap** (resample training seeds, then test episodes within each seed), with 10 000 resamples. One-sided bootstrap p = (1 + #{resampled effect on the wrong side of the threshold}) / (1 + 10 000).
- The confirmatory family {H1a, H1b, H1c, H2a, H2b, H3a, H3b, H4a, H4c, H4d, H5a, H5b, H5d} uses **Holm–Bonferroni** at α = 0.05. H2c is a deterministic all-seeds criterion, reported separately. Threshold hypotheses (H1c, H2a, H3a, H3b, H4c) test against their thresholds.
- Everything else is exploratory and labelled as such. Negative results are reported.

## Theory (stated before experiments)

1. Exact Bayes risk of in-context prediction for linear-Gaussian systems, and its t → ∞ law t·excess → n·p·σ² under ergodic excitation.
2. Closed-loop non-identifiability: under any exploration-free linear feedback u = Kx, the posterior precision along directions orthogonal to range([I; K]) stays at the prior for every t.
3. Construction: two attention layers (a previous-token head, then a moment-accumulating linear-attention layer with p heads) place the exact normalised sufficient statistics in the residual stream.
