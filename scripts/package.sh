#!/bin/bash
# Build the submission package:
#   submission/supplement.zip   pre-registration, deviations, literature check, code, tests, result summaries, pilot logs
#   submission/arxiv_source.zip LaTeX source + class + bib + bbl + figures (what arXiv wants)
set -e
cd "$(dirname "$0")/.."
(cd paper && tectonic -X compile --keep-intermediates main.tex > /dev/null)
mkdir -p submission
rm -f submission/supplement.zip submission/arxiv_source.zip
python - <<'EOF'
import glob, os, zipfile
with zipfile.ZipFile("submission/supplement.zip", "w", zipfile.ZIP_DEFLATED) as z:
    files = ["PREREGISTRATION.md", "docs/DEVIATIONS.md", "docs/LITERATURE.md", "README.md", "LICENSE", "requirements.txt",
             "results/SUMMARY.md", "results/summary.json"]
    files += glob.glob("suss/*.py") + glob.glob("scripts/*.py") + glob.glob("scripts/*.sh") + glob.glob("tests/*.py") + glob.glob("results/pilots/*.log")
    for f in files:
        z.write(f, f)
with zipfile.ZipFile("submission/arxiv_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("paper/*.tex") + glob.glob("paper/*.cls") + glob.glob("paper/*.bib") + glob.glob("paper/*.bbl") + glob.glob("paper/figs/*.pdf"):
        z.write(f, os.path.relpath(f, "paper"))
print("ok")
EOF
cp paper/main.pdf submission/main.pdf
ls -la submission
