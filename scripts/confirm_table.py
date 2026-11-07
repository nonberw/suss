"""Appendix table for pre-registration 2 (results/confirm/summary.json) -> paper/confirm_table.tex."""
import json
import os

ROWS = {
    "H6a": r"KV-only patching of the late blocks: $\tau\ge 0.8$",
    "H6b": r"KV-only patching of block 1: $\tau\le 0.3$",
    "H6c": r"fast weights vs.\ best residual readout ($d_{\rm model}$ dims): gap $\ge 0.15$",
    "H7a": r"prior shift: $J$ closer to training-prior ridge than to new-prior Bayes",
    "H7b": r"prior shift: $J$ closer to training-prior ridge than to OLS",
}
L = [r"\begin{table}[h]", r"\centering\footnotesize",
     r"\caption{Pre-registration 2 (tag \texttt{prereg2-94b6cf8}), registered before these analyses ran. H6: one-sided "
     r"episode-level bootstrap per model; H7: one-sided paired $t$-test over 10 training seeds (difference in mean squared "
     r"distance of the $A$ block, $t{=}8,\dots,47$). Holm correction over all 11 tests.}\label{tab:confirm}",
     r"\begin{tabular}{p{4.7cm}lrrc}", r"\toprule", r"test & model & estimate [95\% CI] & $p_{\rm Holm}$ & holds \\", r"\midrule"]
if os.path.exists("results/confirm/summary.json"):
    R = json.load(open("results/confirm/summary.json"))
    for t in R["tests"]:
        h, _, model = t["id"].partition("|")
        model = model or (f"10 seeds ({t['n_pos']}/{t['n']})" if "n_pos" in t else "")
        fmt = "{:.4f}" if h.startswith("H7") else "{:.2f}"
        est = f"{fmt.format(t['est'])} [{fmt.format(t['ci'][0])}, {fmt.format(t['ci'][1])}]"
        p = r"$<10^{-3}$" if t["p_holm"] < 1e-3 else "{:.3f}".format(t["p_holm"])
        L.append(f"{h}: {ROWS[h]} & {model.replace('_', chr(92) + '_')} & {est} & {p} & {'yes' if t['supported'] else 'no'} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/confirm_table.tex", "w").write("\n".join(L) + "\n")
print("paper/confirm_table.tex written")
