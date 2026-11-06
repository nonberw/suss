# SUSS

**What algorithm does a transformer invent when it has to suss out unknown physics from a few seconds of history?**

You hand a small transformer 48 steps of a system it has never seen. Unknown mass, unknown friction, unknown which way the motor even pushes. It has to predict what happens next, or keep the thing from flying off into the void, with no gradient steps, purely in-context. The vibe is "it's giving system identification". This repo opens the model up and checks whether the vibe is real.

A pre-registered, mechanistic study, trained from scratch on a laptop GPU (RTX 3050 Ti, 4 GB). Paper: [`paper/main.pdf`](paper/main.pdf) (L4DC 2027 format). Hypotheses were committed before any main experiment ran: [`PREREGISTRATION.md`](PREREGISTRATION.md). Everything we changed afterwards, including two bugs we found in ourselves, is in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## TL;DR

Four things we found. Numbers are from `results/summary.json`, 3 training seeds each, 4096 test episodes, hierarchical bootstrap, Holm correction over the 13 pre-registered tests.

**1. It is Bayes-optimal until it isn't.** On linear systems with unknown (A, B) the transformer tracks the *exact* Bayes predictor while data are scarce (within 1.23x up to t = 4), then flattens at an approximation floor: 1.59x the Bayes excess risk averaged over contexts (pre-registered tolerance was 1.25x, so that hypothesis is rejected). Our theorem predicts t x (Bayes excess) -> n p sigma^2 = 0.06; the measured value at t = 47 is 0.057.

**2. The pre-registered mechanism is wrong, and we say so.** We bet the model writes its parameter estimate into the residual stream. It does not. The best linear probe gets R^2 = 0.19 (threshold 0.8), steering along the probe does nothing (slope -0.00003), and swapping the probe subspace between episodes transfers 0.0001 of the behaviour. Linear decodability told us nothing about causal use, three times in a row.

**3. The estimate lives in the keys and values.** Exploratory follow-up, labelled as such: the estimate is spread over the attention key-value memory and applied to the query by attention. The per-head fast weights decode the Bayes estimate with R^2 = 0.89 (residual stream: at most 0.14). The model's input Jacobian *is* the Bayes estimate (R^2 = 0.92). And the causal test: replace only the context keys and values of the last two blocks with another episode's, touch nothing at the query, and 91-94% of the other episode's behaviour transfers. Same story for MuJoCo physics: 1/m is readable from key-value memory with R^2 = 0.97, from activations with 0.32-0.44. If you probe an RMA-style policy's activations for "the latent mass" and find nothing, this is probably why.

**4. The in-context controller learns to explore, then crashes into the dual effect's other wall.** Trained only on closed-loop cost (after an imitation warm start), it gathers 4.2 nats more information about the plant than certainty equivalence without dither, which gathers exactly zero, as our closed-loop identity says it must. It matches dithered CE and Thompson sampling in median cost (1.69x oracle vs 1.57x and 1.78x). But it fails on 10% of plants against 1.4%, almost all of them open-loop unstable (27% failure rate there vs 2-3% elsewhere), and fine-tuning on log-cost makes those failures more severe. Honest verdict: it explores, it is competitive in the typical case, and it is not safe on unstable plants.

Also in the paper: a pushed block with Coulomb friction where the transformer beats physics-feature RLS by 3.3x, a bouncing ball, a scale study (longer training, deeper, wider, a 4-state/2-input system), and three theorems with proofs (exact Bayes risk of in-context prediction under any non-anticipating policy; closed-loop non-identifiability equals excitation energy; a two-layer construction that gathers the sufficient statistics).

## What's in the box

| Experiment | Question | Reference point |
|---|---|---|
| **A** linear systems, unknown (A, B) | Is the transformer Bayes-optimal? What is inside it? | Exact Bayes posterior mean (Monte Carlo over a truncated prior), RLS, OLS, 1-step GD |
| **B** MuJoCo pendulum, pushed block with Coulomb friction, bouncing ball | What does it do when no optimal algorithm is known? | True-parameter sim, nominal sim, physics-feature RLS, no-context MLP |
| **C** in-context adaptive control | Does a controller trained on cost learn to *explore*? | Oracle LQR, best fixed gain, certainty equivalence (CE), CE + dither, explore-then-commit, Thompson sampling |
| **D** mechanism | Probes, steering, activation patching, Jacobians, fast weights, KV-only patching, head roles | Pre-registered thresholds, hierarchical bootstrap, Holm correction |

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q tests
```

```bash
bash scripts/run_all.sh
```

`run_all.sh` runs everything in order with every seed fixed, writes `results/summary.json` and `results/SUMMARY.md` (the confirmatory table), generates `paper/numbers.tex` (every number quoted in the paper is a macro computed from the results; nothing is typed by hand), draws the figures, and builds the PDF with Tectonic. Budget about a day on a 4 GB laptop GPU. It is slower if you also train three other projects on the same laptop and let it fall asleep twice. Ask us how we know.

## Repo map

```
suss/            model (GPT with probe/steer/patch hooks), exact estimators, MuJoCo systems, control, stats
scripts/         train_*, eval_*, mech_*, scale_a, analyze, make_numbers, make_figures, run_all.sh
tests/           20 tests: estimators, theory identities, hooks, KV patching, LQR, MuJoCo, stats
results/         summary.json, SUMMARY.md, pilot logs (large raw outputs are regenerated, not committed)
paper/           L4DC 2027 paper: main.tex -> main.pdf, numbers.tex (generated), figs/
docs/            LITERATURE.md (novelty check, every reference verified), DEVIATIONS.md, SUBMISSION.md
```

## License

MIT. Go wild.
