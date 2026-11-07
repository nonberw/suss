"""Table for the scale study from results/scale/*.json -> paper/scale_table.tex (rows missing results are skipped)."""
import json
import os

import numpy as np

rows = [("base (A, seed 0--2)", None), ("long60k", "same model, 60k steps"), ("small64", "$d{=}64$, 2 layers"),
        ("deep8", "8 layers"), ("wide256", "$d{=}256$"), ("n4m2_s0", "$n{=}4,m{=}2$, seed 0"),
        ("n4m2_s1", "$n{=}4,m{=}2$, seed 1"), ("n4m2_s2", "$n{=}4,m{=}2$, seed 2")]
lines = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Scale study (exploratory, one seed unless stated). Ratio: excess risk relative to the Bayes (ridge) predictor, mean over $t{=}4..47$ and at $t{=}47$. $R^2(J)$: agreement of the query Jacobian with the ridge estimate, mean over $t\in\{4,8,16,32,47\}$.}\label{tab:scale}",
         r"\begin{tabular}{llrrrr}", r"\toprule", r"model & change & params & ratio ($t\ge4$) & ratio ($t{=}47$) & $R^2(J)$ \\", r"\midrule"]
S = json.load(open("results/summary.json")) if os.path.exists("results/summary.json") else {}
if "H1c" in S:
    jac = np.mean([np.mean(S["A_mech3"][s]["jac_r2_bayes_by_t"][4:]) for s in S["A_mech3"]]) if "A_mech3" in S else float("nan")
    c = S["A_curves"]
    lines.append(f"A (3 seeds) & reference, 20k steps & 0.8M & {S['H1c']['point']:.2f} & {c['tf'][47] / c['bayes'][47]:.2f} & {jac:.2f} \\\\")
for tag, change in rows[1:]:
    f = f"results/scale/{tag}.json"
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    tag_tex = tag.replace("_", "\\_")
    lines.append(f"{tag_tex} & {change} & {r['params'] / 1e6:.1f}M & {r['ratio_mean_t4']:.2f} & {r['ratio_t47']:.2f} & {np.mean(list(r['jac_r2'].values())):.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/scale_table.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
