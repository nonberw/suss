"""Seed-level results table (appendix) from results/summary.json -> paper/seed_table.tex."""
import json

import numpy as np

S = json.load(open("results/summary.json"))
L = [r"\begin{table}[h]", r"\centering\small",
     r"\caption{Seed-level results. A: excess-risk ratio to Bayes (mean over $t{=}4..47$) and residual-probe $R^2$ for $\hat\theta$ at the best layer. C: median and mean normalised cost, failure rate and information after 8 steps, for the imitation-only warm start and the cost-fine-tuned controller.}\label{tab:seeds}",
     r"\begin{tabular}{lrr|rrrr|rrrr}", r"\toprule",
     r"& \multicolumn{2}{c|}{A} & \multicolumn{4}{c|}{C, imitation only} & \multicolumn{4}{c}{C, imitation + cost} \\",
     r"seed & ratio & $R^2(\hat\theta)$ & median & mean & fail & info & median & mean & fail & info \\", r"\midrule"]
cs = S.get("C_summary", {})
for s in range(3):
    ratio = float(np.mean(np.array(S["A_curves"]["tf_seed"][s][4:]) / np.array(S["A_curves"]["bayes"][4:]))) if "A_curves" in S else float("nan")
    r2 = max(v for k, v in S["notes"][f"A_seed{s}"]["r2"].items() if k.startswith("theta_bayes@")) if "notes" in S else float("nan")
    bc, tf = cs.get(f"bc_seed{s}", {}), cs.get(f"tf_seed{s}", {})
    f = lambda d, k, fmt: (fmt.format(d[k] * (100 if k == "fail" else 1)) if k in d else "--")
    L.append(f"{s} & {ratio:.2f} & {r2:.2f} & {f(bc, 'median', '{:.2f}')} & {f(bc, 'mean', '{:.0f}')} & {f(bc, 'fail', '{:.1f}')}\\% & {f(bc, 'info8', '{:.1f}')} "
             f"& {f(tf, 'median', '{:.2f}')} & {f(tf, 'mean', '{:.0f}')} & {f(tf, 'fail', '{:.1f}')}\\% & {f(tf, 'info8', '{:.1f}')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/seed_table.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
