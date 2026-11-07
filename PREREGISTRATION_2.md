# Pre-registration 2: confirmatory tests of the key-value account

Registered 2026-11-07, before any of the analyses below was run. The first pre-registration
(`PREREGISTRATION.md`, tag `prereg-d134c5a`) stays as it was.

## Why

In v0.1 the pre-registered mechanism (a linearly readable, causally used parameter estimate in the residual
stream) was rejected. Exploratory analyses then located the sufficient statistics of the context in the
attention key-value (KV) memory of the last two blocks of the main model (4 blocks, d_model = 128), and showed
that the query Jacobian dy/dz tracks the Bayes estimate. Those analyses were run after the account was formed.
This document fixes two new tests of the account in advance: (A) on architectures it was not derived from, and
(B) a prediction about which algorithm the network runs, under a prior it was not trained on.

What has already been seen and is therefore not tested here: the behaviour (excess-risk ratio) and the query
Jacobian of the scale-study models (`results/summary.json`, key `scale`); the in-distribution query Jacobian of
the main model, which departs from the Bayes estimate for t < 8 (seen while building the browser demo). No KV
patching, fast-weight probe or prior-shift analysis has been run on any model named below.

## Part A: the KV account on other architectures

**Models** (trained for the scale study, `runs/scale/`): `long60k` (4 blocks, d_model 128, 60k steps), `deep8`
(8 blocks, d_model 128), `wide256` (4 blocks, d_model 256). `small64` (2 blocks, d_model 64) is reported
descriptively, without hypotheses.

**Data and procedure**, identical to the exploratory analyses (`scripts/mech_a4.py`, `scripts/mech_a5.py`): the
4096 reference episodes of `results/a/references.pt`; probes fit on the first 2048 and evaluated on the other
2048; patching on the held-out half with pairs (B, A'), where A' is the next held-out episode's context followed
by B's query token. Transfer tau = mean over pairs of (y_patched - y_B) . (y_A' - y_B) / |y_A' - y_B|^2 at t = 47.
"Late blocks" = the second half of the blocks: blocks 3-4 of the 4-block models, blocks 5-8 of `deep8`.

- **H6a (localisation).** Replacing only the context keys and values of the late blocks transfers tau >= 0.8.
- **H6b (specificity).** Replacing only the context keys and values of block 1 transfers tau <= 0.3.
- **H6c (readability at matched dimension).** The probe for the Bayes estimate theta_47 from the fast weights
  Cov(v, k) of the best late block, randomly projected (Gaussian, fixed seed) to d_model dimensions, has held-out
  R^2 at least 0.15 above the best residual-stream readout at d_model dimensions (residual at the query position
  or mean-pooled over positions, best block). Ridge probes in dual form, alpha by 5-fold CV, as in `mech_a5.py`.

Each hypothesis is tested in each of the three models (9 tests).

## Part B: which algorithm, under a prior shift

**Account under test.** The network applies the Bayes (ridge) estimate for its *training* prior, computed from
the statistics of the context. Within one episode it cannot know that the population of systems has changed, so
on systems from a different prior its effective estimate should follow ridge with the training prior, not the
Bayes estimate for the new prior, and not unregularised least squares.

**Test systems.** 1024 episodes, T = 48, drawn with a seed fixed in the code: A entries ~ N(0, 0.1^2) instead of
N(0, 0.4^2); B entries ~ N(0, 0.5^2) as in training; truncation rho(A) < 0.95, noise sigma = 0.1, u ~ N(0, 1),
x_0 ~ N(0, I) as in training. These systems lie inside the support of the training prior.

**Estimators** from the pairs s < t: ridge with the training prior (lambda = sigma^2 / sd^2 = 0.0625, 0.0625,
0.04), the Bayes estimate for the new prior (ridge with lambda = 1, 1, 0.04), and OLS (pseudo-inverse).

**Network estimate.** The query Jacobian J_t = dy_t / dz_t, exact by automatic differentiation.

**Metric.** D_X = mean over episodes and over t = 8, ..., 47 of ||J_t[:, A] - theta_X,t[:, A]||_F^2, the A block
only (where the priors differ). t starts at 8 because the Jacobian departs from the Bayes estimate before that
even in distribution.

- **H7a.** D_(training-prior ridge) < D_(new-prior Bayes).
- **H7b.** D_(training-prior ridge) < D_(OLS).

**Unit of analysis**: the 10 training seeds of the main model (`runs/a/seed0-9.pt`); one-sided paired t-test
across seeds on the differences, with the number of seeds in the predicted direction reported.

## Statistics and decisions

Part A: one-sided episode-level bootstrap (10,000 resamples of held-out episodes, or of pairs for tau), probes
fixed at their fitted weights. Part B: one-sided paired t-test across the 10 seeds. alpha = 0.05 with
Holm-Bonferroni over all 11 tests. A hypothesis is called supported for the KV account if it holds in every model
it is tested on. Every result is reported, whichever way it goes, in the paper and in `results/confirm/`.

Exploratory, without hypotheses: `small64`; the B column under the prior shift; behaviour under the prior shift.
