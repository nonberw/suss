# Changelog

Numbers come from `results/summary.json`. Nothing below changes a result unless it says so.

## v0.2.0 (2026-11-07)

- **Second pre-registration** (`PREREGISTRATION_2.md`, tag `prereg2-94b6cf8`), registered and pushed before the analyses ran: the key-value account on long60k, deep8 and wide256 (H6a-c) and a prior-shift test of the algorithm (H7a-b). **All 11 tests hold** (Holm): late-block KV swap 0.85-0.96, block 1 0.01-0.13, readability gap 0.21-0.35 R^2; under a shifted prior the query Jacobian stays closer to ridge with the training prior (0.026) than to the new prior's Bayes estimate (0.028) or OLS (0.036) in 10/10 seeds. Scripts: `confirm_a.py`, `confirm_b.py`, `confirm_stats.py`; results in `results/confirm/`. The paper reports it in Section 8 and Appendix E; main text still 10 pages.
- Fixed: number ranges inside math mode printed as two minus signs ("0.91--0.94" as "0.91 - - 0.94") in 20 places, including the abstract. `make_numbers.py` now emits ranges and minus signs that typeset the same in text and math. No number changed.
- All 51 checkpoints on the Hugging Face Hub: https://huggingface.co/nonberew/suss-models
- **Live demo** in the browser (`docs/demo/`, https://nonberw.github.io/suss/demo/): the experiment-A model (seed 0) in plain JavaScript, next to the exact Bayes estimator, with the query Jacobian shown as the model's own parameter estimate. `tests/test_demo.py` checks the JavaScript forward pass and Jacobian against PyTorch.
- Issue templates ("Challenge a result", bug report), this changelog, `requirements-lock.txt` with the exact versions used, a lint step in CI (`ruff.toml`: syntax errors and undefined names only).
- Hugging Face model card and upload script (`hf/`, `scripts/upload_hf.py`).

## v0.1.3 (2026-11-07)

- All 51 trained checkpoints released (`suss_models.zip`) plus a 3 MB quickstart model; manifest in `docs/MODELS.md`.
- Quickstart notebook (runs in Colab), project page on GitHub Pages, tests in CI.
- Fixed: the paper said "quadrupling the width" for the wide model of the scale study; it doubles the width (4x the parameters). Its number (1.30x) is unchanged. The README and the v0.1.0 notes said 3.3x for the block-vs-RLS ratio; the paper's number is 3.2x.
- Archived on Zenodo: https://doi.org/10.5281/zenodo.24032754

## v0.1.2 (2026-11-07)

- Author name, `CITATION.cff`, `.zenodo.json`. First Zenodo archive: https://doi.org/10.5281/zenodo.24031899. Same results as v0.1.1.

## v0.1.1 (2026-11-07)

- All planned runs complete: the scale study and the ten-seed replication of experiment A. Main text at the 10-page limit.

## v0.1.0 (2026-11-07)

- First preprint: experiments A-C, the mechanism analyses, the pre-registered confirmatory table and the exploratory follow-ups with dimension-matched controls.

## Pre-registration (2026-11-05)

- Hypotheses, metrics, seeds and tests committed before the main experiments: tag `prereg-d134c5a`. Every later change is listed in `docs/DEVIATIONS.md`.
