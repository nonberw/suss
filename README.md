# SUSS

**What does a transformer compute when it has to suss out unknown physics from a few seconds of history?**

You hand a small transformer 48 steps of a system it has never seen. Unknown mass, unknown friction, unknown which way the motor even pushes. It has to predict what happens next, or keep the thing from flying off into the void, with no gradient steps, purely in-context. The vibe is "it's giving system identification". This repo opens the model up and checks whether the vibe is real, and where exactly it lives.

A pre-registered, mechanistic study, trained from scratch on a laptop GPU (RTX 3050 Ti, 4 GB). Paper: [`paper/main.pdf`](paper/main.pdf) (L4DC 2027 format, preprint). Hypotheses were committed before any main experiment ran: [`PREREGISTRATION.md`](PREREGISTRATION.md), tag `prereg-d134c5a`. Everything changed afterwards, including two bugs we found in our own code and an external review we acted on, is in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## TL;DR

Numbers are from `results/summary.json`: 3 pre-registered training seeds per model (10 for the replication of experiment A), 4096 test episodes, hierarchical bootstrap, Holm correction over the 13 pre-registered tests. Anything added after the pre-registration is labelled exploratory, here and in the paper.

**1. Bayes-optimal until it isn't.** On linear systems with unknown (A, B), the transformer tracks the *exact* Bayes predictor while data are scarce (within 1.23x up to t = 4), then flattens at an approximation floor: 1.59x the Bayes excess risk averaged over contexts (pre-registered tolerance was 1.25x, so that hypothesis is rejected). Exploratory scale study: training 3x longer or doubling depth brings it to ~1.22x, 4x width to 1.30x, and a 0.1M-parameter model sits at 5.3x. Still not 1.

**2. The pre-registered mechanism is wrong, and we say so.** We bet the model writes its parameter estimate into the residual stream. It doesn't, at least not linearly or causally: the best linear probe gets R^2 = 0.19 (threshold 0.8), steering along the probe does nothing, and swapping the probe subspace between episodes transfers 0.0001 of the behaviour. Linear decodability told us nothing about causal use.

**3. The sufficient statistics live in the key-value memory.** Exploratory, with the controls a reviewer asked for. The attention fast weights Cov(v, k) of the last two blocks read the normalised sufficient statistics (S, C) with R^2 = 0.92 and the Bayes estimate with 0.90. Squash them to 128 dims and they still read 0.84-0.87; the residual stream tops out at 0.58-0.61 however you read it (query position, all 48 positions, mean-pooled, last context token). Shuffled labels: 0.00. Causally: swap only the context keys and values of blocks 3-4 with another episode's, touch nothing at the query, and 91-94% of the other episode's behaviour transfers. The same swap on block 1 alone: 7-16%. Two heads per block do most of it (0.78-0.82 vs 0.41-0.51 for two random heads). Replicated on 7 more seeds trained after the hypothesis was formed. The model's input Jacobian matches the Bayes estimate (R^2 = 0.92), which is what you'd expect if attention applies the stored statistics to the query. Fun twist: the pre-registered "statistics first, solve later" story was right about the *order* and wrong about the *place*.

**4. Physics.** On a block pushed around with Coulomb friction, the transformer beats linear-in-parameters physics regression by 3.3x and even beats structured system identification with the exact MuJoCo simulator (0.0058 vs 0.011): from (x, xdot) alone the contact state is hidden, and the transformer recovers it from history. On the pendulum, 1/m reads from key-value memory with R^2 = 0.97 vs 0.32-0.44 from activations. On a bouncing ball added later, the story only half-repeats, and we say that too.

**5. The in-context controller learns to explore, then hits the dual effect's other wall.** Trained by imitating an LQR oracle and then on closed-loop cost, it gathers 4.2 nats more information about the plant than certainty equivalence without dither (which gathers nothing about B, as the closed-loop identifiability identity says it must), though 1.4 nats less than dithered CE. Its median cost is competitive (1.69x oracle vs 1.57x dithered CE, 1.78x Thompson sampling), but it fails on 10% of plants vs 1.4%, almost all open-loop unstable. Fine-tuning on raw cost or on CVaR of the worst 10% diverges outright. Honest verdict: it explores, it's fine on a typical plant, and you shouldn't put it on an unstable one.

Plus: a scale study (longer, deeper, a 4-state/2-input system), and the textbook theory restated for closed-loop in-context prediction (Bayes risk, its n p sigma^2 / t law, closed-loop non-identifiability), with proofs, presented as background rather than as a contribution.

## What's in the box

| Experiment | Question | Reference point |
|---|---|---|
| **A** linear systems, unknown (A, B) | Bayes-optimal? What's inside? | Exact Bayes posterior mean (Monte Carlo, truncated prior), RLS, OLS, 1-step GD |
| **B** MuJoCo pendulum, pushed block with Coulomb friction, bouncing ball | What happens when no optimal algorithm is known? | True-parameter sim, structured sysID with the exact simulator, nominal sim, physics-feature RLS, no-context MLP |
| **C** in-context adaptive control | Does a controller trained on cost learn to *explore*? | Oracle LQR, best fixed gain, certainty equivalence (CE), CE + dither, explore-then-commit, Thompson sampling |
| **D** mechanism | Probes (dimension-matched), steering, activation patching, Jacobians, fast weights, KV-only patching, head roles | Pre-registered thresholds, hierarchical bootstrap, Holm; seed-level replication |

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

`run_all.sh` runs everything in order with every seed fixed, writes `results/summary.json` and `results/SUMMARY.md` (the confirmatory table), generates `paper/numbers.tex` (every number in the paper is a macro computed from the results, nothing is typed by hand), draws the figures, builds the PDF with Tectonic and packs `submission/`. Budget about a day on a 4 GB laptop GPU. It's slower if you also train three other projects on the same laptop and let it fall asleep twice. Ask us how we know.

## Repo map

```
suss/            model (GPT with probe/steer/patch hooks), exact estimators, MuJoCo systems, control, stats
scripts/         train_*, eval_*, mech_*, sysid_block, scale_a, analyze, make_numbers, make_figures, run_all.sh
tests/           20 tests: estimators, theory identities, hooks, KV patching, LQR, MuJoCo, stats
results/         summary.json, SUMMARY.md, pilot logs (large raw outputs are regenerated, not committed)
paper/           the paper: main.tex -> main.pdf, numbers.tex (generated), figs/
docs/            LITERATURE.md (novelty check, every reference verified), DEVIATIONS.md, SUBMISSION.md
```

## Status

Preprint (v0.1.x). Not peer reviewed. Target venue: L4DC 2027; see `docs/SUBMISSION.md`. Single author, with heavy AI assistance (disclosed in the paper). Issues and "you're wrong because" welcome.

## License

MIT. Go wild.
