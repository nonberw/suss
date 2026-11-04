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
