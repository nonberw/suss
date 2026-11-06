"""Table of the controller-training pilots (seed 99, never reported as results) from results/pilots/*.log."""
import re

rows = [
    ("pilot_c_norm_3e-4.log", r"BPTT, $J/J_{\rm oracle}$, lr $3{\cdot}10^{-4}$"),
    ("pilot_c_lognorm_3e-4.log", r"BPTT, $\log(J/J_{\rm oracle})$, lr $3{\cdot}10^{-4}$"),
    ("pilot_c_lognorm_1e-4_sq1_tb0.log", r"BPTT, $\log$, squashed inputs$^\dagger$"),
    ("pilot_c_lognorm_1e-4_sq1_tb8.log", r"same, 8-step truncated BPTT"),
    ("pilot_bcft.log", r"DAgger warm start (3000) + fine-tune (1500)$^\dagger$"),
    ("final_warmstart_raw_seed0.log", r"final warm start (6000) + raw cost, seed 0"),
    ("final_warmstart_cvar_seed0.log", r"final warm start (6000) + CVaR$_{0.9}$, seed 0"),
]
pat = re.compile(r"step\s+(\d+) loss \S+ val\(mean,median,fail\) \(([\d.e+-]+), ([\d.e+-]+), ([\d.e+-]+)(?:, [\d.e+-]+)?\)")
lines = [r"\begin{table}[h]", r"\centering\small", r"\caption{Controller-training pilots (seed 99, 1024 held-out plants). Cost is $J/J_{\rm oracle}$; ``fail'' means $J/J_{\rm oracle}{>}10$. $^\dagger$Intended lr $10^{-4}$, but a bug ran these pilots at $3{\cdot}10^{-4}$. The final runs use $10^{-4}$.}\label{tab:pilots}",
         r"\begin{tabular}{lrrrr}", r"\toprule", r"variant & step & median & mean & fail \\", r"\midrule"]
for f, lab in rows:
    try:
        vals = [m.groups() for m in map(pat.search, open(f"results/pilots/{f}")) if m]
    except FileNotFoundError:
        continue
    keep = vals if f == "pilot_bcft.log" else [v for v in vals if v[0] != "0"]  # step 0 of fine-tuning = warm start only
    if f.startswith("final_"):
        keep = [v for v in vals if v[0] in ("0", "500", "1000", "2999")]
    for i, (st, mean, med, fail) in enumerate(keep):
        lines.append(f"{lab if i == 0 else ''} & {st} & {float(med):.2f} & {float(mean):.0f} & {100 * float(fail):.0f}\\% \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("paper/pilot_table.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
