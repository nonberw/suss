# Changelog

Numbers come from `results/summary.json`. Nothing below changes a result unless it says so.

## Unreleased

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
