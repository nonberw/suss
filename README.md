# SUSS

**What algorithm does a transformer invent when it has to suss out unknown physics from a few seconds of history?**

You hand a small transformer 48 steps of a system it has never seen. Unknown mass, unknown friction, unknown which way the motor even pushes. It has to predict what happens next, or keep the thing from flying off into the void. It does this with no gradient steps, purely in-context. The vibe is "it's giving system identification". This repo checks whether that vibe is real by opening the model up and poking it.

A pre-registered, mechanistic study, trained from scratch on a laptop GPU (RTX 3050 Ti, 4 GB). Paper: [`paper/main.pdf`](paper/main.pdf). Hypotheses were committed before any main experiment ran: [`PREREGISTRATION.md`](PREREGISTRATION.md). Everything we changed afterwards is in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## TL;DR

HEADLINE_RESULTS

## What's in the box

| Experiment | Question | Reference point |
|---|---|---|
| **A** linear systems, unknown (A, B) | Is the transformer Bayes-optimal? What is inside it? | Exact Bayes posterior mean (Monte Carlo over a truncated prior), RLS, OLS, 1-step GD |
| **B** MuJoCo pendulum + pushed block with Coulomb friction | What does it do when no optimal algorithm is known? | True-parameter sim, nominal sim, physics-feature RLS, no-context MLP |
| **C** in-context adaptive control | Does a controller trained only on cost learn to *explore*? | Oracle LQR, best fixed gain, certainty equivalence (CE), CE + dither, explore-then-commit |
| **D** mechanism | Probes, steering, activation patching, head roles | Pre-registered thresholds, hierarchical bootstrap, Holm correction |

Theory (in the paper):
1. An exact Bayes-risk identity for in-context prediction under any non-anticipating policy, plus a parameter-free context-length law `t * risk -> n p sigma^2`.
2. Closed-loop non-identifiability: the information you gain along the unexcited directions is exactly the excitation energy you put in. Zero dither means zero learning, forever.
3. A two-layer construction that gathers the exact sufficient statistics.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

```bash
bash scripts/run_all.sh
```

`run_all.sh` runs everything in order, with every seed fixed:
- train A (3 seeds), evaluate, probe and patch;
- generate the MuJoCo datasets, train B (TF + MLP, 2 systems x 3 seeds), evaluate and probe;
- train C (3 seeds plus fixed gains), tune the baselines and evaluate;
- `scripts/analyze.py`, which does the pre-registered stats and writes `results/SUMMARY.md`;
- `scripts/make_figures.py`, which writes the figures;
- `tectonic paper/main.tex`, which builds the PDF.

Total compute is a few hours on a 4 GB laptop GPU. It is slower if you are also training three other projects on the same laptop. Ask us how we know.

## Repo map

```
suss/            model (GPT with probe/steer/patch hooks), exact estimators, MuJoCo systems, control, stats
scripts/         train_*, eval_*, mech_*, analyze.py, make_figures.py
tests/           unit tests for estimators, theory identities, hooks, LQR, MuJoCo, stats
results/         summary.json, SUMMARY.md (all confirmatory tests), per-experiment outputs
paper/           L4DC 2027 formatted paper (main.tex -> main.pdf)
docs/            literature/novelty check, deviations log, submission checklist
```

## License

MIT. Go wild.
