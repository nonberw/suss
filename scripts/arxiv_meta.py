"""Write submission/ARXIV.md: copy-paste metadata for the arXiv form, with the abstract expanded to plain text
(macros from paper/numbers.tex substituted, LaTeX stripped). arXiv limits abstracts to 1920 characters."""
import re

macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open("paper/numbers.tex", encoding="utf-8").read()))
s = open("paper/abstract.tex", encoding="utf-8").read()
for _ in range(3):
    s = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", lambda m: macros.get(m.group(1), "\\" + m.group(1)), s)
rules = [(r"\\mbox\{([^}]*)\}", r"\1"), (r"\\ensuremath\{([^}]*)\}", r"\1"), (r"\$\\pm\$", "+-"), (r"\\pm", "+-"), (r"\$\\times\$", "x"), (r"\\times", "x"), (r"\{=\}", " = "), (r"\{\\le\}", " <= "),
         (r"\\le\b", "<="), (r"--", "-"), (r"\\%", "%"), (r"\\emph\{([^}]*)\}", r"\1"), (r"\$", ""), (r"\\,", " "),
         (r"R\^2", "R^2"), (r"\\ ", " "), (r"~", " "), (r"\s+", " ")]
for a, b in rules:
    s = re.sub(a, b, s)
abstract = s.strip()
assert "\\" not in abstract, abstract
title = "What a Transformer Learns When It Adapts to Unknown Physics In-Context: Probing, Patching and Controlling"
try:
    import pymupdf
    pages = len(pymupdf.open("submission/main_arxiv_preview.pdf"))
except Exception:
    pages = "about 19"
md = f"""# arXiv submission metadata (copy-paste)

**Upload:** `submission/arxiv_source.zip` (LaTeX source; arXiv compiles `main.tex` with pdfLaTeX).

**Title:** {title}

**Authors:** Anuar Zhambakin

**Abstract** ({len(abstract)} characters; arXiv limit 1920):

{abstract}

**Comments:** {pages} pages (10 main + references and appendix), 4 figures. Code, pre-registration and data: https://github.com/nonberw/suss (doi:10.5281/zenodo.24031898)

**Primary category:** cs.LG (Machine Learning). **Cross-lists:** eess.SY (Systems and Control), cs.RO (Robotics).

**License:** CC BY 4.0 (recommended for open research; choose another if you prefer).

**Journal-ref / DOI:** leave empty (not published yet).
"""
open("submission/ARXIV.md", "w", encoding="utf-8").write(md)
print(md)
