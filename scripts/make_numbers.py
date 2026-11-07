"""Write paper/numbers.tex: every number quoted in the paper, as LaTeX macros computed from results/summary.json.
Missing results render as '??' so nothing is ever typed by hand."""
import json
import re

import numpy as np

S = json.load(open("results/summary.json"))
M = {}


def tidy(s):
    """Plain numbers and ranges ('a--b') must typeset the same in text and in math: '--' in math mode would print two
    minus signs. Ranges get an en-dash in an \\mbox, negative numbers a real minus; equal ends collapse to one number."""
    m = re.fullmatch(r"(-?[\d.]+)(?:--(-?[\d.]+))?", s)
    if not m:
        return s
    ends = [e for e in m.groups() if e is not None]
    if len(ends) == 2 and ends[0] == ends[1]:
        ends = ends[:1]
    return r"\mbox{--}".join(e.replace("-", r"\ensuremath{-}", 1) if e.startswith("-") else e for e in ends)


def put(name, fn, fmt="{:.2f}"):
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = fn()
        if not isinstance(v, str) and (v != v):  # nan
            raise ValueError("missing")
        M[name] = tidy(v if isinstance(v, str) else fmt.format(v))
    except Exception:
        M[name] = "??"


def ci(key, fmt="{:.2f}"):
    return lambda: "[" + fmt.format(S[key]["ci"][0]) + ", " + fmt.format(S[key]["ci"][1]) + "]"


def mean_seeds(d, key):
    return np.mean([d[s][key] for s in d])


def rng_seeds(d, key, fmt="{:.2f}"):
    v = [d[s][key] for s in d]
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else lo + "--" + hi


# ---- A, confirmatory
put("AdOLS", lambda: S["H1a"]["point"], "{:.3f}"); put("AdOLSci", ci("H1a", "{:.3f}"))
put("AdGD", lambda: S["H1b"]["point"], "{:.3f}"); put("AdGDci", ci("H1b", "{:.3f}"))
put("Aratio", lambda: S["H1c"]["point"]); put("Aratioci", ci("H1c"))
put("AtBayes", lambda: 47 * S["A_curves"]["bayes"][47], "{:.3f}")
put("AtTheory", lambda: 47 * S["A_curves"]["theory_ridge"][47], "{:.3f}")
put("Ashort", lambda: max(S["A_curves"]["tf"][t] / S["A_curves"]["bayes"][t] for t in range(1, 5)))
put("Afloor", lambda: (S["A_curves"]["tf"][47] - S["A_curves"]["bayes"][47]) * 1e3, "{:.1f}")
put("ARtheta", lambda: S["H2a"]["point"]); put("ARthetaci", ci("H2a"))
put("AdR", lambda: S["H2b"]["point"], "{:.3f}"); put("AdRci", ci("H2b", "{:.3f}"))
put("Amom", lambda: max(np.mean(np.array(S["A_mech"]["r2_layers"]["moments"]), 0)))
put("Alogtr", lambda: max(np.mean(np.array(S["A_mech"]["r2_layers"]["post_logtrace"]), 0)))
put("Aslope", lambda: S["H3a"]["point"], "{:.4f}"); put("Ar", lambda: S["H3a"]["r"], "{:.3f}")
put("Atausub", lambda: S["H3b"]["point"], "{:.4f}"); put("Atausubci", ci("H3b", "{:.4f}"))
put("Ataurand", lambda: S["A_mech"]["tau_rand"], "{:.3f}")
put("AtauLastThree", lambda: "{:.2f}--{:.2f}".format(*sorted([x[3] for x in S["A_mech"]["tau_last"]])[::2]))
put("AtauCtxTwo", lambda: min(x[2] for x in S["A_mech"]["tau_ctx"]))
put("Aprev", lambda: "{:.2f}--{:.2f}".format(min(np.max(np.array(p)) for p in S["A_mech"]["prev_token"]), max(np.max(np.array(p)) for p in S["A_mech"]["prev_token"])))
put("AaccZero", lambda: S["A_bayes_acceptance"][0]); put("AaccLast", lambda: S["A_bayes_acceptance"][1], "{:.3f}")
put("Aeta", lambda: S["A_gd1_eta"])
# ---- A, exploratory mechanism
m2 = S.get("A_mech2", {})
put("AyhatLthree", lambda: rng_seeds({s: {"v": m2[s]["lin_yhat"][3]} for s in m2}, "v"))
put("AyhatLfour", lambda: rng_seeds({s: {"v": m2[s]["lin_yhat"][4]} for s in m2}, "v"))
put("AmlpTheta", lambda: np.mean([max(m2[s]["mlp_theta"]) for s in m2]))
put("AsteerYhat", lambda: np.mean([m2[s]["steer_yhat"][4][0] for s in m2]))
put("AtauYhat", lambda: np.mean([m2[s]["tau_yhat_sub"][4] for s in m2]))
m3 = S.get("A_mech3", {})
put("AjacBayes", lambda: rng_seeds(m3, "jac_r2_bayes")); put("AjacTrue", lambda: rng_seeds(m3, "jac_r2_true"))
put("AjacOLS", lambda: rng_seeds(m3, "jac_r2_ols"))
put("AcfLin", lambda: rng_seeds(m3, "cf_r2_linear_jac")); put("AcfSlope", lambda: rng_seeds(m3, "cf_slope_bayes"))
put("AcfBayes", lambda: rng_seeds(m3, "cf_r2_bayes"))
put("AfwR", lambda: rng_seeds({s: {"v": m3[s]["fw_r2"][-1]} for s in m3}, "v"))
put("AfwRthree", lambda: rng_seeds({s: {"v": m3[s]["fw_r2"][-2]} for s in m3}, "v"))
put("AresidR", lambda: rng_seeds({s: {"v": max(m3[s]["resid_r2"])} for s in m3}, "v"))
m4 = S.get("A_mech4", {})
put("AkvThreeFour", lambda: rng_seeds(m4, "kv-2-3")); put("AkvTwoFour", lambda: rng_seeds(m4, "kv-1-2-3"))
put("AkOnly", lambda: rng_seeds(m4, "k-2-3")); put("AvOnly", lambda: rng_seeds(m4, "v-2-3"))
put("AkvOne", lambda: rng_seeds(m4, "kv-0"))
# ---- B
for name, P in (("block", "K"), ("pendulum", "P"), ("bouncer", "Q")):
    for k, lab in (("tf", "TF"), ("feature_rls", "RLS"), ("nominal", "Nom"), ("oracle", "Orc"), ("mlp", "MLP"), ("persistence", "Per")):
        def f(name=name, k=k):
            d = S[f"B_{name}_mse"]
            if k in ("tf", "mlp"):
                return np.mean([d[f"{k}_err_seed{s}"] for s in range(3)])
            return d[f"{k}_err"]
        put(f"B{P}{lab}", f, "{:.2g}")
put("BKratio", lambda: S["B_block_mse"]["feature_rls_err"] / np.mean([S["B_block_mse"][f"tf_err_seed{s}"] for s in range(3)]), "{:.1f}")
put("BPratio", lambda: S["B_pendulum_mse"]["nominal_err"] / np.mean([S["B_pendulum_mse"][f"tf_err_seed{s}"] for s in range(3)]), "{:.0f}")
put("HfourA", lambda: S["H4a"]["point"], "{:.4f}"); put("HfourAci", ci("H4a", "{:.4f}"))
for name, P, tg in (("pendulum", "P", "g"), ("pendulum", "P", "inv_m"), ("pendulum", "P", "c_over_m"), ("block", "K", "mu"), ("block", "K", "inv_m"),
                    ("bouncer", "Q", "g"), ("bouncer", "Q", "inv_m"), ("bouncer", "Q", "zeta")):
    put(f"Bprobe{P}{tg.replace('_', '')}", lambda name=name, tg=tg: "{:.2f}--{:.2f}".format(
        min(S[f"B_{name}_probe"][f"seed{s}"][tg][0] for s in range(3)), max(S[f"B_{name}_probe"][f"seed{s}"][tg][0] for s in range(3))))
for name, P in (("pendulum", "P"), ("block", "K")):
    put(f"Bsteer{P}", lambda name=name: S[f"H4d_{name}"]["frac_correct"] * 100, "{:.0f}")
put("BsteerQ", lambda: S["B_bouncer_steer"]["frac_correct"] * 100, "{:.0f}")
for name, P in (("pendulum", "P"), ("block", "K"), ("bouncer", "Q")):
    put(f"Bfunc{P}", lambda name=name: "{:.2f}--{:.2f}".format(min(S[f"B_{name}_functional_invm"]["corr"]), max(S[f"B_{name}_functional_invm"]["corr"])))
bm = S.get("B_mech2", {})


def bfw(name, tg, which):
    vals = []
    for s in range(3):
        best = max((bm[k][which] for k in bm if k.startswith(f"{name}|{s}|") and k.endswith(f"|{tg}")), default=np.nan)
        vals.append(best)
    if any(np.isnan(v) for v in vals):
        raise ValueError("missing")
    lo, hi = "{:.2f}".format(min(vals)), "{:.2f}".format(max(vals))
    return lo if lo == hi else lo + "--" + hi


for name, P, tg in (("pendulum", "P", "inv_m"), ("pendulum", "P", "g"), ("block", "K", "mu"), ("block", "K", "inv_m"), ("bouncer", "Q", "inv_m"), ("bouncer", "Q", "zeta")):
    put(f"Bfw{P}{tg.replace('_', '')}", lambda name=name, tg=tg: bfw(name, tg, 0))
    put(f"Bres{P}{tg.replace('_', '')}", lambda name=name, tg=tg: bfw(name, tg, 1))
# ---- C
cs = S.get("C_summary", {})
for k, lab in (("ce", "CE"), ("ce_dither", "CED"), ("etc", "ETC"), ("ts", "TS")):
    put(f"C{lab}mean", lambda k=k: cs[k]["mean"]); put(f"C{lab}med", lambda k=k: cs[k]["median"])
    put(f"C{lab}fail", lambda k=k: cs[k]["fail"] * 100, "{:.1f}"); put(f"C{lab}info", lambda k=k: cs[k]["info8"], "{:.1f}")
for k, lab in (("ce", "CE"), ("ce_dither", "CED"), ("etc", "ETC"), ("ts", "TS")):
    put(f"C{lab}cvar", lambda k=k: cs[k]["cvar95"], "{:.1f}"); put(f"C{lab}pninety", lambda k=k: cs[k]["p90"], "{:.1f}")
put("CTFcvar", lambda: np.mean([cs[f"tf_seed{s}"]["cvar95"] for s in range(3)]), "{:.0f}")
put("CTFpninety", lambda: np.mean([cs[f"tf_seed{s}"]["p90"] for s in range(3)]), "{:.1f}")
put("CBCcvar", lambda: np.mean([cs[f"bc_seed{s}"]["cvar95"] for s in range(3)]), "{:.0f}")
put("CBCpninety", lambda: np.mean([cs[f"bc_seed{s}"]["p90"] for s in range(3)]), "{:.1f}")
put("CTFmean", lambda: np.mean([cs[f"tf_seed{s}"]["mean"] for s in range(3)]))
put("CTFmed", lambda: np.mean([cs[f"tf_seed{s}"]["median"] for s in range(3)]))
put("CTFfail", lambda: np.mean([cs[f"tf_seed{s}"]["fail"] for s in range(3)]) * 100, "{:.1f}")
put("CTFinfo", lambda: np.mean([cs[f"tf_seed{s}"]["info8"] for s in range(3)]), "{:.1f}")
put("CFIXmean", lambda: np.mean([cs[f"fixed_seed{s}"]["mean"] for s in range(3)]), "{:.0f}")
put("CFIXmed", lambda: np.mean([cs[f"fixed_seed{s}"]["median"] for s in range(3)]))
put("CBCmean", lambda: np.mean([cs[f"bc_seed{s}"]["mean"] for s in range(3)]))
put("CBCmed", lambda: np.mean([cs[f"bc_seed{s}"]["median"] for s in range(3)]))
put("CBCfail", lambda: np.mean([cs[f"bc_seed{s}"]["fail"] for s in range(3)]) * 100, "{:.1f}")
put("CBCinfo", lambda: np.mean([cs[f"bc_seed{s}"]["info8"] for s in range(3)]), "{:.1f}")
put("CbcMinusFt", lambda: S["C_bc_minus_ft"]["point"]); put("CbcMinusFtci", ci("C_bc_minus_ft"))
put("CinfoFtBc", lambda: S["C_info8_ft_minus_bc"]["point"]); put("CinfoFtBcci", ci("C_info8_ft_minus_bc"))
put("CORCinfo", lambda: S["C_oracle_info8"], "{:.1f}")
put("CprobeB", lambda: np.mean([max(v for k, v in S["C_probe"][f"seed{s}"].items() if k.startswith("B_true@")) for s in range(3)]))
put("CprobeBhat", lambda: np.mean([max(v for k, v in S["C_probe"][f"seed{s}"].items() if k.startswith("B_hat@")) for s in range(3)]))
put("Cflip", lambda: np.mean(S["C_steer_flip"]) * 100, "{:.0f}")
fd = S.get("C_fail_diag", {})
put("CfracUnst", lambda: fd["frac_unstable"] * 100, "{:.0f}")
for k, lab in (("ce_dither", "CED"), ("ts", "TS"), ("etc", "ETC")):
    put(f"CfailUnst{lab}", lambda k=k: fd["by_controller"][k]["fail_unstable"] * 100, "{:.1f}")
    put(f"CfailStab{lab}", lambda k=k: fd["by_controller"][k]["fail_stable"] * 100, "{:.1f}")
for k, lab in (("tf", "TF"), ("bc", "BC")):
    put(f"CfailUnst{lab}", lambda k=k: np.mean([fd["by_controller"][f"{k}_seed{s}"]["fail_unstable"] for s in range(3)]) * 100, "{:.0f}")
    put(f"CfailStab{lab}", lambda k=k: np.mean([fd["by_controller"][f"{k}_seed{s}"]["fail_stable"] for s in range(3)]) * 100, "{:.1f}")
put("CTFmedRange", lambda: "{:.2f}--{:.2f}".format(min(cs[f"tf_seed{s}"]["median"] for s in range(3)), max(cs[f"tf_seed{s}"]["median"] for s in range(3))))
put("CTFfailRange", lambda: "{:.0f}--{:.0f}".format(100 * min(cs[f"tf_seed{s}"]["fail"] for s in range(3)), 100 * max(cs[f"tf_seed{s}"]["fail"] for s in range(3))))
put("CTFmeanRange", lambda: "{:.0f}--{:.0f}".format(min(cs[f"tf_seed{s}"]["mean"] for s in range(3)), max(cs[f"tf_seed{s}"]["mean"] for s in range(3))))
for h in ("H5a", "H5b", "H5d"):
    put(h.replace("5", "five"), lambda h=h: S[h]["point"]); put(h.replace("5", "five") + "ci", ci(h))
for k, lab in (("ce_dither", "CED"), ("etc", "ETC"), ("ts", "TS")):
    put(f"Cvs{lab}", lambda k=k: S[f"H5c_{k}_minus_tf"]["point"]); put(f"Cvs{lab}ci", ci(f"H5c_{k}_minus_tf"))
    put(f"Cvs{lab}med", lambda k=k: S[f"H5c_{k}_minus_tf_median"]["point"]); put(f"Cvs{lab}medci", ci(f"H5c_{k}_minus_tf_median"))
    put(f"Cwins{lab}", lambda k=k: S[f"H5c_{k}_tf_wins"] * 100, "{:.0f}")
    put(f"Cinfo{lab}", lambda k=k: S[f"C_info8_tf_minus_{k}"]["point"]); put(f"Cinfo{lab}ci", ci(f"C_info8_tf_minus_{k}"))
mc = S.get("C_mech", {})
put("CmechGainEnd", lambda: rng_seeds({s: {"v": mc[s]["gain_r2_oracle"][-1]} for s in mc}, "v"))
put("CmechGainMid", lambda: rng_seeds({s: {"v": mc[s]["gain_r2_oracle"][2]} for s in mc}, "v"))
put("CmechGainErrEnd", lambda: np.mean([mc[s]["gain_err_oracle"][-1] for s in mc]))
put("CmechCEErrEnd", lambda: np.mean([mc[s]["ce_err_oracle"][-1] for s in mc]))
put("CmechGainErrEarly", lambda: np.mean([mc[s]["gain_err_oracle"][0] for s in mc]), "{:.1f}")
put("CmechCEErrEarly", lambda: np.mean([mc[s]["ce_err_oracle"][0] for s in mc]), "{:.1f}")
put("CmechBfwEnd", lambda: rng_seeds({s: {"v": mc[s]["probe_B_fw"]["47"]} for s in mc}, "v"))
put("CmechBfwEarly", lambda: rng_seeds({s: {"v": mc[s]["probe_B_fw"]["8"]} for s in mc}, "v"))
put("CmechBresEnd", lambda: rng_seeds({s: {"v": mc[s]["probe_B_resid"]["47"]} for s in mc}, "v"))
put("CmechBresEarly", lambda: rng_seeds({s: {"v": mc[s]["probe_B_resid"]["8"]} for s in mc}, "v"))
m5 = S.get("A_mech5", {})


def m5rng(fn, tn):
    d = m5[f"{fn}|{tn}"]
    lo, hi = "{:.2f}".format(d["min"]), "{:.2f}".format(d["max"])
    return lo if lo == hi else lo + "--" + hi


for fn, lab in (("resid_q", "Q"), ("resid_ctx", "Ctx"), ("ctx_rp128", "CtxRp"), ("fw", "Fw"), ("fw_rp128", "FwRp"), ("fw_pca128", "FwPca")):
    for tn, tl in (("theta_bayes", "Theta"), ("moments", "Mom"), ("theta_true", "True")):
        put(f"Ac{lab}{tl}", lambda fn=fn, tn=tn: m5rng(fn, tn))
    put(f"Ac{lab}Dim", lambda fn=fn: m5[f"{fn}|dim"], "{:d}")
def m5blk(fn, tn, blk):
    v = [bb[blk] for bb in m5[f"{fn}|{tn}"]["by_block"]]
    lo, hi = "{:.2f}".format(min(v)), "{:.2f}".format(max(v))
    return lo if lo == hi else lo + "--" + hi


put("AcFwMomBtwo", lambda: m5blk("fw", "moments", 1)); put("AcFwThetaBtwo", lambda: m5blk("fw", "theta_bayes", 1))
put("AcFwMomBone", lambda: m5blk("fw", "moments", 0)); put("AcFwThetaBone", lambda: m5blk("fw", "theta_bayes", 0))
m6 = S.get("A_mech6", {})


def rng_list(v, fmt="{:.2f}"):
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else lo + "--" + hi


put("AcShuffled", lambda: rng_list(m6["shuffled_max"]))
put("AcMeanTheta", lambda: rng_list(m6["resid_mean|theta_bayes"])); put("AcLastTheta", lambda: rng_list(m6["resid_last_ctx|theta_bayes"]))
put("AcMeanMom", lambda: rng_list(m6["resid_mean|moments"])); put("AcLastMom", lambda: rng_list(m6["resid_last_ctx|moments"]))
put("AcCeilLin", lambda: m6["ceiling_linear"]); put("AcCeilMlp", lambda: m6["ceiling_mlp"])
put("ARemarkR", lambda: rng_list(m6["remark_r2_blocks34"]))
put("ANsharp", lambda: rng_list(m6["n_sharp_blocks34"], "{:d}"))
put("ARemarkSharp", lambda: rng_list(m6["remark_r2_sharp"], "{:.0f}"))
put("AcResBestTheta", lambda: rng_list([max(m5["resid_q|theta_bayes"]["by_block"][i][l] for l in range(4)) if False else max(
    max(m5["resid_q|theta_bayes"]["by_block"][i]), max(m5["resid_ctx|theta_bayes"]["by_block"][i]), m6["resid_mean|theta_bayes"][i], m6["resid_last_ctx|theta_bayes"][i]) for i in range(3)]))
put("AcResBestMom", lambda: rng_list([max(max(m5["resid_q|moments"]["by_block"][i]), max(m5["resid_ctx|moments"]["by_block"][i]), m6["resid_mean|moments"][i], m6["resid_last_ctx|moments"][i]) for i in range(3)]))
put("ARemarkRone", lambda: rng_list([b[0] for b in m6["remark_r2_by_block"]]))
put("ATopTwo", lambda: rng_list(m6["top2"])); put("ARandTwo", lambda: rng_list(m6["random2"]))
put("ATopThree", lambda: rng_list(m6["top3"])); put("ARandThree", lambda: rng_list(m6["random3"]))
ar = S.get("A_rep", {})


def repfmt(key, fmt="{:.2f}"):
    d = ar[key]
    return (fmt + r" $\pm$ " + fmt).format(d["mean"], d["sd"])


put("RepN", lambda: ar["ratio"]["n"], "{:d}")
put("RepRatio", lambda: repfmt("ratio")); put("RepKv", lambda: repfmt("kv34")); put("RepFwTheta", lambda: repfmt("fw_theta"))
put("RepFwMom", lambda: repfmt("fw_moments")); put("RepResQ", lambda: repfmt("resid_q_theta")); put("RepResCtx", lambda: repfmt("resid_ctx_theta"))
put("RepKvMin", lambda: ar["kv34"]["min"]); put("RepFwMin", lambda: ar["fw_theta"]["min"]); put("RepResMax", lambda: max(ar["resid_q_theta"]["max"], ar["resid_ctx_theta"]["max"]))
# ---- scale study
sc = S.get("scale", {})
put("ScBaseRatio", lambda: S["H1c"]["point"])
put("ScLongRatio", lambda: sc["long60k"]["ratio_mean_t4"]); put("ScLongRatioEnd", lambda: sc["long60k"]["ratio_t47"])
put("ScDeepRatio", lambda: sc["deep8"]["ratio_mean_t4"]); put("ScWideRatio", lambda: sc["wide256"]["ratio_mean_t4"])
put("ScSmallRatio", lambda: sc["small64"]["ratio_mean_t4"])
put("ScFourRatio", lambda: "{:.2f}--{:.2f}".format(min(sc[f"n4m2_s{s}"]["ratio_mean_t4"] for s in range(3)), max(sc[f"n4m2_s{s}"]["ratio_mean_t4"] for s in range(3))))
put("ScFourJac", lambda: "{:.2f}--{:.2f}".format(min(np.mean(list(sc[f"n4m2_s{s}"]["jac_r2"].values())) for s in range(3)), max(np.mean(list(sc[f"n4m2_s{s}"]["jac_r2"].values())) for s in range(3))))
put("ScFourJacEnd", lambda: "{:.2f}--{:.2f}".format(min(sc[f"n4m2_s{s}"]["jac_r2"]["47"] for s in range(3)), max(sc[f"n4m2_s{s}"]["jac_r2"]["47"] for s in range(3))))
put("ScLongJacEnd", lambda: sc["long60k"]["jac_r2"]["47"])
put("ScFourTR", lambda: "{:.2f}--{:.2f}".format(min(sc[f"n4m2_s{s}"]["t_excess_ridge_47"] for s in range(3)), max(sc[f"n4m2_s{s}"]["t_excess_ridge_47"] for s in range(3))))
put("ScFourNP", lambda: sc["n4m2_s0"]["npsigma2"])
put("Hours", lambda: 15, "{:.0f}")  # estimate: sum of per-job times at full GPU speed (A 0.6 h, B 3.2 h, C 4.8 h, scale 4.5 h, evaluations 1 h); wall-clock was far longer on the shared GPU
put("BKSys", lambda: S["B_block_mse"]["sysid_err"], "{:.2g}")
put("BKOrcObs", lambda: S["B_block_mse"]["oracle_obs_err"], "{:.2g}")
put("BKSysNote", lambda: ("better than the transformer" if S["B_block_mse"]["sysid_err"] < np.mean([S["B_block_mse"][f"tf_err_seed{s}"] for s in range(3)]) else "worse than the transformer")
    + "; with the true parameters and the same observation-only state it gets {:.2g}".format(S["B_block_mse"]["oracle_obs_err"]))
put("AratioEnd", lambda: S["A_curves"]["tf"][47] / S["A_curves"]["bayes"][47])
put("ScDeepJacEnd", lambda: sc["deep8"]["jac_r2"]["47"])
put("ScSmallJacEnd", lambda: sc["small64"]["jac_r2"]["47"])
put("ScMinBigJacEnd", lambda: min(sc[k]["jac_r2"]["47"] for k in ("long60k", "deep8", "wide256")))
put("ScWideJacEnd", lambda: sc["wide256"]["jac_r2"]["47"])
put("ScFourFinite", lambda: sc["n4m2_s0"]["npsigma2"] * 47 / (47 - (sc["n4m2_s0"]["args"]["n"] + sc["n4m2_s0"]["args"]["m"]) - 1), "{:.3f}")
put("ScFourTR", lambda: "{:.3f}".format(np.mean([sc[f"n4m2_s{s}"]["t_excess_ridge_47"] for s in range(3) if f"n4m2_s{s}" in sc])))
put("ScFourRatio", lambda: rng_list([sc[f"n4m2_s{s}"]["ratio_mean_t4"] for s in range(3) if f"n4m2_s{s}" in sc], "{:.1f}"))
put("ScFourJacEnd", lambda: rng_list([sc[f"n4m2_s{s}"]["jac_r2"]["47"] for s in range(3) if f"n4m2_s{s}" in sc]))
# ---- pre-registration 2 (results/confirm/summary.json)
try:
    CF = json.load(open("results/confirm/summary.json"))
except FileNotFoundError:
    CF = {"tests": [], "a": {}, "b_per_seed": {}, "verdict": {}}
CFT = {t["id"]: t for t in CF["tests"]}
CFM = ("long60k", "deep8", "wide256")
put("CfTauLate", lambda: rng_list([CF["a"][m]["tau_late"] for m in CFM]))
put("CfTauOne", lambda: rng_list([CF["a"][m]["tau_block1"] for m in CFM]))
put("CfKv", lambda: rng_list([CF["a"][m]["r2_best_kv"] for m in CFM]))
put("CfRes", lambda: rng_list([CF["a"][m]["r2_best_resid"] for m in CFM]))
put("CfGap", lambda: rng_list([CF["a"][m]["gap"] for m in CFM]))
put("CfSmallTauLate", lambda: CF["a"]["small64"]["tau_late"]); put("CfSmallGap", lambda: CF["a"]["small64"]["gap"])
put("CfNsix", lambda: sum(CFT[f"{h}|{m}"]["supported"] for h in ("H6a", "H6b", "H6c") for m in CFM), "{:d}")
for h in ("H7a", "H7b"):
    put(f"Cf{h[1:].replace('7', 'Seven')}N", lambda h=h: CFT[h]["n_pos"], "{:d}")
    put(f"Cf{h[1:].replace('7', 'Seven')}P", lambda h=h: ("<10^{-3}" if CFT[h]["p_holm"] < 1e-3 else "={:.3f}".format(CFT[h]["p_holm"])))
for k in ("ridge_train", "bayes_new", "ols"):
    put("CfD" + k.replace("_", "").replace("ridgetrain", "Train").replace("bayesnew", "New").replace("ols", "Ols"),
        lambda k=k: np.mean([v[k] for v in CF["b_per_seed"].values()]), "{:.3f}")
# ---- Holm
for k, v in S.get("holm", {}).items():
    put("P" + k.replace("1", "one").replace("2", "two").replace("3", "three").replace("4", "four").replace("5", "five"),
        lambda v=v: ("<10^{-3}" if v["p_holm"] < 1e-3 else "={:.3f}".format(v["p_holm"])))
out = "% AUTO-GENERATED by scripts/make_numbers.py from results/summary.json. Do not edit.\n"
out += "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n"
open("paper/numbers.tex", "w").write(out)
print(f"{len(M)} macros, {sum(v == '??' for v in M.values())} missing")
