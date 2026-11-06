# Deviations from the pre-registration

Every change made after the pre-registration commit (`d134c5a`), with its reason. None of them was made after seeing a confirmatory result, unless that is stated explicitly.

1. **Excess-risk estimator (A).** The pre-registration defined excess(t) = MSE(t) − nσ². We compute E‖ŷ − θz_t‖² against the noiseless conditional mean instead. Both have the same expectation, but ours has lower variance. The decision was made while writing the evaluation code, before any evaluation ran.
2. **Information-gain prior (C1, H5d).** Λ₀ in log det(Λ₀ + σ⁻² S₈) was left unspecified. We use Λ₀ = I. It is a fixed, policy-independent constant, so the paired comparison is unaffected in direction.
3. **Unspecified training details.**
   - Fixed constant-gain controller: Adam, lr 0.02, 1000 steps, batch 1024, cost clamped at 1e4 per plant for stable gradients.
   - No-context MLP (B): 2 hidden layers of width 256, 5000 steps.
   - Neither was given in the pre-registration.
4. **Incremental forward for the controller (C1).** Back-propagating through 48 full forward passes ran out of the 4 GB of GPU memory. We added a KV-cached incremental forward. It is numerically equivalent to the full forward (max abs diff < 1e-6; tested in `tests/test_core.py`).
5. **CE baseline at t = 0.** With the prior mean B = 0, certainty equivalence outputs u ≡ 0 and never excites the input. That is the behaviour of the pre-registered baseline, not a change, but it makes "CE (no dither)" a degenerate baseline. We report it as pre-registered and treat the tuned CE+dither and explore-then-CE as the meaningful comparisons (H5c).
6. **Pendulum signal size (B).** A pilot (seed 99, not reported) showed that nominal-parameter simulation already gets 0.5% normalised error on the pendulum: parameter effects over 0.05 s are small. We kept the pre-registered design unchanged. As a result, the pendulum is mostly a check on the true-model-class case, and the block carries the contact result.
7. **Exploratory follow-up for A (added after seeing seed-0 mechanistic results).** The pre-registered probes found θ̂ weakly decodable (R² ≈ 0.2), and steering/patching along them had no effect. We therefore added `scripts/mech_a2.py`: probes for the query-specific Bayes prediction, MLP probes, position-specific probes, a logit lens, and patching/steering along the prediction subspace. All of it is labelled exploratory, and the pre-registered tests are reported unchanged.
8. **Controller training uses CUDA-graph replay.** The machine was shared with other GPU jobs, which made eager-mode closed-loop BPTT take more than 3 s/step. The same loss, optimiser and schedule are captured once as a CUDA graph and replayed. Fresh plants are sampled each step into static buffers. Same algorithm, different execution.
9. **Fixed-controller seeds 1–2 were trained on CPU** (identical code, `--device cpu`) because of GPU contention.
10. **Exploratory mechanism analyses for A and B** (added after the pre-registered probes failed):
    - query-Jacobian analysis (`scripts/mech_a3.py`);
    - key–value fast-weight probes (`mech_a3.py`, `mech_b2.py`);
    - KV-only activation patching (`mech_a4.py`);
    - functional 1/m readout from the model's input sensitivity (`analyze.py`).
    They are motivated by Remark 1 in the paper, which was written after the probes failed, and are reported as exploratory.
11. **Third MuJoCo system, bouncing ball (exploratory).** Pilots were used to tune it before any training: smaller physics timestep and a damping-ratio range of [0.2, 1.0], because low damping ratios made MuJoCo's soft contact inject energy; and forcing kept below the ball's weight. Its feature-RLS baseline uses a linear spring–damper contact model.
12. **Thompson-sampling baseline for C1 (exploratory).** A classical Bayesian dual-control heuristic, tuned on the same tuning plants as the other baselines.
13. **Scale study (exploratory, `scripts/scale_a.py`).** Longer training (60k steps), deeper (8 layers), wider (d=256) and smaller (d=64, 2 layers) models, plus a larger system (n=4, m=2, 24 unknown parameters). The reference there is ridge (= Bayes up to the truncation of the prior).
14. **Controller training (C1).** Pure closed-loop BPTT on the cost, as pre-registered, did not train reliably in pilots (seed 99, 2000 steps; validation on 1024 held-out plants, normalised cost J/J_oracle):

    | Variant | Validation result |
    |---|---|
    | raw cost | loss oscillated between 25 and 4700 |
    | J/J_oracle | median 85, 96% failures |
    | log(J/J_oracle), lr 3e-4 | median 2.4 at step 1000, then collapsed to median 11.7 and 52% failures |
    | same with input squashing (intended lr 1e-4, actually 3e-4, see item 15) | 98% failures |
    | with truncated BPTT (8 steps) | median 4.3, 36% failures at step 1000 |

    The cause is exploding gradients through open-loop-unstable plants. The final controller therefore uses a DAgger warm start (`scripts/train_c2.py`), imitating the oracle LQR action from its own history, followed by BPTT fine-tuning on the cost (`train_c.py --init`). Both stages are evaluated, so the paper can say what cost-driven fine-tuning adds to imitation.
15. **Learning-rate bug (found 2026-11-06 04:40).** In the CUDA-graph branch of `train_c.py`, the schedule called `lr_at(step, steps)` without the `--lr` argument, so every pilot labelled "lr 1e-4" and the first (aborted) fine-tuning attempt actually ran at peak lr 3e-4. The bug is fixed. The final fine-tuning runs use the intended 1e-4, and the pilot table marks the affected pilots.
16. **Checkpoint selection for the controller.** The final controller is the fine-tuning checkpoint with the lowest validation log-cost on 512 held-out plants (seed 777, disjoint from the test plants), evaluated every 500 steps. The warm start itself (before any fine-tuning) is also a candidate, so fine-tuning cannot make the reported controller worse on validation. The last checkpoint is saved too (`seed*_last.pt`). In-training validation uses 512 plants instead of 1024 to save GPU time; it is used for monitoring and selection only.
17. **Changes after an external review of the draft (2026-11-06 evening).** Added, all exploratory and labelled as such in the paper:
    - dimension-matched probe controls (`scripts/mech_a5.py`): residual stream over all context positions, random projections and PCA of the fast weights to 128 dimensions, and a direct comparison of "estimate" vs "sufficient statistics" as probe targets;
    - a structured system-identification baseline for the block with the exact simulator and the true model class (`scripts/sysid_block.py`);
    - controller objective pilots from the same warm start: raw cost and CVaR (worst 10%), seed 0, 3000 steps (`--loss raw|cvar`), evaluated on the same test plants (`scripts/eval_c_variants.py`);
    - a seed-level results table and a note that the cluster bootstrap over three seeds under-estimates between-seed variance.
    Text changes: the title no longer claims the estimate "lives in the keys" (keys or values alone transfer only part of the behaviour); the theory section is presented as background rather than contribution; the statement of the closed-loop proposition now distinguishes on-policy from off-policy queries; the comparison of the controller's information gain is made primarily against dithered CE rather than against exploration-free CE; and claims of out-of-sample confirmation were removed until the corresponding results exist.
