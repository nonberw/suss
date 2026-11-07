#!/bin/bash
# Build the submission package:
#   submission/supplement.zip   pre-registration, deviations, literature check, code, tests, result summaries, pilot logs
#   submission/arxiv_source.zip LaTeX source + class + bib + bbl + figures (what arXiv wants)
set -e
cd "$(dirname "$0")/.."
# tectonic: use the one on PATH, else the known local install
TECTONIC=$(command -v tectonic || echo "$HOME/tools/tectonic/tectonic.exe")
(cd paper && "$TECTONIC" -X compile --keep-intermediates main.tex > /dev/null)
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
        arc = os.path.relpath(f, "paper").replace(os.sep, "/")
        if arc == "main.tex":  # arXiv version: "Preprint" header instead of the PMLR proceedings header
            src = open(f, encoding="utf-8").read().replace("\\documentclass{l4dc2027}", "\\def\\arxivpreprint{}\n\\documentclass{l4dc2027}", 1)
            assert "arxivpreprint{}" in src
            z.writestr(arc, src)
        else:
            z.write(f, arc)
print("ok")
EOF
cp paper/main.pdf submission/main.pdf
python scripts/arxiv_meta.py > /dev/null
# check that the arXiv variant builds (XeTeX here; arXiv itself uses pdfLaTeX)
rm -rf submission/arxiv_check && mkdir -p submission/arxiv_check && (cd submission/arxiv_check && unzip -q ../arxiv_source.zip && "$TECTONIC" -X compile main.tex > /dev/null && cp main.pdf ../main_arxiv_preview.pdf) && rm -rf submission/arxiv_check
ls -la submission
