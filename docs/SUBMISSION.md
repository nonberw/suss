# Submission checklist (L4DC 2027)

**Venue:** 9th Annual Learning for Dynamics & Control Conference (L4DC 2027), KTH Stockholm, June 16–18, 2027.
**Paper deadline: November 13, 2026 (firm, will not be extended).** Decisions are due Jan 25, 2027, and camera-ready is due Apr 19, 2027.
Source: https://l4dc2027.control.ee.ethz.ch/ and its `/author-information` page, checked 2026-11-05.

Format rules (from the author-information page and the official template):
- at most 10 PMLR pages including figures and tables; references and acknowledgments are extra;
- single-blind, so author names stay on;
- English PDF submitted on OpenReview;
- an optional supplement (proofs, extra experiments), uploaded with the paper; reviewers judge only the 10-page paper;
- a disclosure of generative AI use (the paper has a "Generative AI use" section, as the template asks).

## What you (the author) must do

1. **OpenReview account.** Create or log in at https://openreview.net. A profile can take a few days to be approved if you use a non-institutional email, so **do this first**.
2. **Check the authors line** in `paper/main.tex` (`\coltauthor{...}`, marked `TODO(author)`): L4DC is single-blind, so put your real name, an email and an affiliation (or "Independent researcher"). The `\Name{}` macro of the jmlr class needs at least two words, so `\Name{First Last}` works and a single-word handle does not. Rebuild with `cd paper && tectonic main.tex`.
3. Go to https://openreview.net/group?id=L4DC.org/2027 and click the submission link.
4. Run `bash scripts/package.sh` (or `scripts/run_all.sh`, which ends with it). It writes `submission/main.pdf`, `submission/supplement.zip` (pre-registration, deviations log, literature check, code, tests, result summaries, pilot logs) and `submission/arxiv_source.zip`. Upload `submission/main.pdf` as the paper and `submission/supplement.zip` as the supplementary material.
5. Fill in the title, abstract (copy it from the PDF), keywords and subject areas (learning-based control, system identification).
6. Submit **before Nov 13, 2026**. The time zone is not stated on the site, so treat it as "end of Nov 12 AoE" to be safe.
7. Optional, recommended after submission: arXiv. You need an arXiv account and an endorsement for `cs.LG` / `eess.SY` (first-time submitters often need one). Upload `submission/arxiv_source.zip` (LaTeX source + figures + .bbl). L4DC is single-blind, so an arXiv preprint is fine.

## Backup venues if L4DC is missed or the paper is rejected

- **ICML 2027.** Expected deadline late January 2027; check icml.cc when the call appears. It needs the ICML style and 8 pages, so the paper would need reformatting.
- **NeurIPS 2027.** Usually mid-May 2027.
- **RSS 2027.** One aggregator lists Dec 4, 2026, but this is unverified; check roboticsconference.org.
- Workshops: NeurIPS / ICML workshops on in-context learning or interpretability (calls come out roughly two months before each conference).
