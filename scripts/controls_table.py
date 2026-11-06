"""Dimension-matched probe controls (appendix table) from results/a/mech5.pt -> paper/controls_table.tex."""
import os

import numpy as np
import torch

FEATS = [("resid_q", "residual stream, query position"), ("resid_ctx", "residual stream, all 48 positions"),
         ("ctx_rp128", "random projection of the above to 128"), ("fw", "key--value fast weights $\\widehat{\\mathrm{Cov}}(v,k)$"),
         ("fw_rp128", "random projection of fast weights to 128"), ("fw_pca128", "top-128 PCs of fast weights")]
TG = [("theta_bayes", "$\\hat\\theta_{47}$"), ("moments", "$(S,C)_{47}$"), ("theta_true", "$\\theta$")]
L = [r"\begin{table}[h]", r"\centering\small",
     r"\caption{Dimension-matched probe controls at $t{=}47$ (held-out $R^2$, best block, min--max over 3 seeds). Ridge probes in dual form with 5-fold CV. $(S,C)$ are the normalised sufficient statistics.}\label{tab:controls}",
     r"\begin{tabular}{lr" + "r" * len(TG) + "}", r"\toprule", "features & dim & " + " & ".join(t for _, t in TG) + r" \\", r"\midrule"]
if os.path.exists("results/a/mech5.pt"):
    M = torch.load("results/a/mech5.pt", weights_only=False)
    for fn, lab in FEATS:
        cells = []
        for tn, _ in TG:
            v = [max(M[s][(fn, l, tn)] for l in range(4)) for s in M]
            lo, hi = "{:.2f}".format(min(v)), "{:.2f}".format(max(v))
            cells.append(lo if lo == hi else f"{lo}--{hi}")
        dim = M[0][("dim", fn, 0)]
        L.append(f"{lab} & {dim} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/controls_table.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
