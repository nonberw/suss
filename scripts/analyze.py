"""Pre-registered confirmatory analysis: hierarchical bootstrap + Holm-Bonferroni.
Reads results/{a,b,c}, writes results/summary.json and results/SUMMARY.md."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss.probes import r2_from_parts
from suss.stats import ci, hier_boot, holm, p_greater, p_less

NB = 10_000
SEEDS = [0, 1, 2]
rng = np.random.default_rng(12345)
out, pvals, notes = {}, {}, {}


def boot_custom(n_per_seed, fn, n_boot=NB):
    """fn(list of (seed, idx)) -> statistic; hierarchical resampling of seeds then episodes."""
    S = len(n_per_seed)
    point = fn([(s, np.arange(n_per_seed[s])) for s in range(S)])
    dist = np.empty(n_boot)
    for b in range(n_boot):
        sel = [(s, rng.integers(0, n_per_seed[s], n_per_seed[s])) for s in rng.integers(0, S, S)]
        dist[b] = fn(sel)
    return point, dist


def rec(name, point, dist, p, desc):
    lo, hi = ci(dist)
    out[name] = dict(point=float(point), ci=[lo, hi], p=float(p), desc=desc)
    pvals[name] = p


# ---------------- Experiment A
if os.path.exists("results/a/eval_seed0.npz"):
    ev = [np.load(f"results/a/eval_seed{s}.npz") for s in SEEDS]
    ref = np.load("results/a/references_excess.npz")
    for key, h in (("ols", "H1a"), ("gd1", "H1b")):
        vals = [(e[f"dist_{key}"][:, 1:] - e["dist_bayes"][:, 1:]).mean(1) for e in ev]
        pt, d = hier_boot(vals, n_boot=NB)
        rec(h, pt, d, p_greater(d), f"mean_t d(TF,{key}) - d(TF,Bayes) > 0")
    tf = [e["tf_excess"] for e in ev]
    bay = ref["bayes"]

    def ratio(sel):
        num = np.concatenate([tf[s][i][:, 4:] for s, i in sel]).mean(0)
        den = np.concatenate([bay[i][:, 4:] for s, i in sel]).mean(0)
        return float((num / den).mean())
    pt, d = boot_custom([len(x) for x in tf], ratio)
    rec("H1c", pt, d, p_less(d, 1.25), "mean_t excess_TF/excess_Bayes < 1.25 (t=4..47)")
    Rref = torch.load("results/a/references.pt", map_location="cpu", weights_only=False)
    out["A_bayes_acceptance"] = [float(Rref["acc"][:, 0].mean()), float(Rref["acc"][:, -1].mean())]
    out["A_gd1_eta"] = float(Rref["eta"])
    out["A_curves"] = dict(
        tf=[float(np.concatenate([x[:, t] for x in tf]).mean()) for t in range(48)],
        **{k: [float(ref[k][:, t].mean()) for t in range(48)] for k in ["bayes", "ridge", "ols", "gd1", "zero", "theory_ridge"]},
        tf_seed=[[float(x[:, t].mean()) for t in range(48)] for x in tf],
    )

if os.path.exists("results/a/mech_seed0.pt"):
    M = [torch.load(f"results/a/mech_seed{s}.pt", weights_only=False) for s in SEEDS]
    for s, m in enumerate(M):
        notes[f"A_seed{s}"] = dict(l_star=m["l_star"], l_mom=m["l_mom"], r2={f"{k[0]}@{k[1]}": round(v, 4) for k, v in m["r2"].items()})
    nE = [len(m["probe"][("theta_bayes", 0)]["cnt"]) for m in M]

    def r2stat(name, sel, layer_key="l_star"):
        return np.mean([r2_from_parts(M[s]["probe"][(name, M[s][layer_key])], i) for s, i in sel])
    pt, d = boot_custom(nE, lambda sel: r2stat("theta_bayes", sel))
    rec("H2a", pt, d, p_greater(d, 0.8), "max_l R2(theta_Bayes) > 0.8")
    pt, d = boot_custom(nE, lambda sel: r2stat("theta_bayes", sel) - r2stat("theta_true", sel))
    rec("H2b", pt, d, p_greater(d), "R2(theta_Bayes) - R2(theta_true) > 0 at l*")
    h2c = all(m["l_mom"] < m["l_star"] for m in M)
    out["H2c"] = dict(holds_all_seeds=bool(h2c), l_mom=[m["l_mom"] for m in M], l_star=[m["l_star"] for m in M],
                      desc="argmax_l R2(moments) < argmax_l R2(theta_Bayes) in all seeds")
    # H3a steering slope and r
    obs = [m["steer"]["obs"] for m in M]  # (K, E, 2)
    pred = [m["steer"]["pred"] for m in M]

    def slope_r(sel):
        o = np.concatenate([obs[s][:, i].ravel() for s, i in sel])
        p = np.concatenate([pred[s][:, i].ravel() for s, i in sel])
        c = np.cov(p, o)
        return c[0, 1] / c[0, 0], c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
    nS = [o.shape[1] for o in obs]
    pt_s, d_s = boot_custom(nS, lambda sel: slope_r(sel)[0], n_boot=2000)
    pt_r, d_r = boot_custom(nS, lambda sel: slope_r(sel)[1], n_boot=2000)
    p3a = max(p_greater(d_s, 0.5), p_less(d_s, 1.5), p_greater(d_r, 0.7))
    rec("H3a", pt_s, d_s, p3a, "steering slope in [0.5,1.5] and r >= 0.7")
    out["H3a"]["r"] = float(pt_r); out["H3a"]["r_ci"] = list(ci(d_r))
    pt, d = hier_boot([m["tau_sub"] for m in M], n_boot=NB)
    rec("H3b", pt, d, p_greater(d, 0.5), "subspace-patching transfer > 0.5")
    out["A_mech"] = dict(
        tau_rand=float(np.mean([m["tau_rand"].mean() for m in M])),
        tau_ctx=[[float(x.mean()) for x in m["tau_ctx"]] for m in M],
        tau_last=[[float(x.mean()) for x in m["tau_last"]] for m in M],
        tau_head=[m["tau_head"].tolist() for m in M],
        entropy=[m["entropy"].tolist() for m in M], prev_token=[m["prev_token"].tolist() for m in M],
        ablation=[m["ablation"].tolist() for m in M], r2_by_t=[m["r2_by_t"] for m in M],
        r2_layers={name: [[m["r2"][(name, l)] for l in range(5)] for m in M]
                   for name in ["theta_true", "theta_bayes", "theta_ols", "moments", "moments_S", "moments_C", "post_logtrace"]},
    )

# ---------------- Experiment B
if os.path.exists("results/b/block_eval.npz"):
    sl = slice(24, 48)
    for name in ["block", "pendulum"]:
        E = np.load(f"results/b/{name}_eval.npz")
        vals = [(E["feature_rls_err"][:, sl] - E[f"tf_err_seed{s}"][:, sl]).mean(1) for s in SEEDS]
        pt, d = hier_boot(vals, n_boot=NB)
        if name == "block":
            rec("H4a", pt, d, p_greater(d), "block: MSE(feature-RLS) - MSE(TF) > 0, t=24..47")
        else:
            out["H4b_pendulum_rls_minus_tf"] = dict(point=float(pt), ci=list(ci(d)), desc="exploratory; negative = RLS better")
        out[f"B_{name}_mse"] = {k: float(E[k][:, sl].mean()) for k in E.files if "_err" in k}
        out[f"B_{name}_vel"] = {k: float(E[k][:, sl].mean()) for k in E.files if "_vel" in k}
        out[f"B_{name}_curves"] = {k: E[k].mean(0).tolist() for k in E.files if "_err" in k}
    pp, dists = [], []
    for name, tgts in (("pendulum", ["g", "inv_m"]), ("block", ["mu"])):
        Mb = torch.load(f"results/b/{name}_mech.pt", weights_only=False)
        out[f"B_{name}_probe"] = {f"seed{s}": {k: [round(v[0], 4), v[1]] for k, v in Mb[f"best_seed{s}"].items()} for s in SEEDS}
        out[f"B_{name}_probe_by_t"] = {k[0]: [Mb[(k[0], "by_t", s)] for s in SEEDS] for k in Mb if isinstance(k, tuple) and len(k) == 3 and k[1] == "by_t" and k[2] == 0}
        for tg in tgts:
            parts = [Mb["parts"][(tg, Mb[f"best_seed{s}"][tg][1], s)] for s in SEEDS]
            n = [len(p["cnt"]) for p in parts]
            pt, d = boot_custom(n, lambda sel: np.mean([r2_from_parts(parts[s], i) for s, i in sel]))
            pp.append(p_greater(d, 0.5)); out[f"H4c_{name}_{tg}"] = dict(point=float(pt), ci=list(ci(d)))
        st = [Mb[f"steer_seed{s}"] for s in SEEDS]
        dsens = [x["steered"] - x["base"] for x in st]
        pt, d = hier_boot(dsens, n_boot=NB)
        out[f"H4d_{name}"] = dict(point=float(pt), ci=list(ci(d)), frac_correct=float(np.mean(np.concatenate(dsens) > 0)),
                                  base_sens=float(np.mean(np.concatenate([x["base"] for x in st]))))
        dists.append(p_greater(d))
    # EXPLORATORY: functional readout. The model's own sensitivity d v'/d u at the last token is its implicit
    # estimate of dt/m (pendulum: dt/(m l^2)); compare to the truth on the evaluation half.
    for name in ["pendulum", "block", "bouncer"]:
        if not os.path.exists(f"results/b/{name}_mech.pt"):
            continue
        Mb = torch.load(f"results/b/{name}_mech.pt", weights_only=False)
        Dp = np.load(f"data/{name}_test.npz")["params"]
        invm = 1 / Dp[len(Dp) // 2:, 0]
        cors, slopes = [], []
        for s_ in SEEDS:
            b = Mb[f"steer_seed{s_}"]["base"]
            cors.append(float(np.corrcoef(b, invm)[0, 1]))
            slopes.append(float(np.polyfit(invm, b, 1)[0]))
        phys = 0.05 / (0.25 if name == "pendulum" else 1.0)
        out[f"B_{name}_functional_invm"] = dict(corr=cors, slope=slopes, physics_slope=phys,
                                               desc="corr(d v'/d u, 1/m) over episodes; physics slope dt/(l^2) or dt")
        if name == "bouncer":
            E_ = np.load("results/b/bouncer_eval.npz")
            out["B_bouncer_mse"] = {k: float(E_[k][:, 24:48].mean()) for k in E_.files if "_err" in k}
            out["B_bouncer_vel"] = {k: float(E_[k][:, 24:48].mean()) for k in E_.files if "_vel" in k}
            out["B_bouncer_curves"] = {k: E_[k].mean(0).tolist() for k in E_.files if "_err" in k}
            out["B_bouncer_probe"] = {f"seed{s_}": {k: [round(v[0], 4), v[1]] for k, v in Mb[f"best_seed{s_}"].items()} for s_ in SEEDS}
            vals = [(E_["nominal_err"][:, 24:48] - E_[f"tf_err_seed{s_}"][:, 24:48]).mean(1) for s_ in SEEDS]
            pt, d = hier_boot(vals, n_boot=NB)
            out["B_bouncer_nominal_minus_tf"] = dict(point=float(pt), ci=list(ci(d)), desc="exploratory")
            st = [Mb[f"steer_seed{s_}"] for s_ in SEEDS]
            dsens = [x["steered"] - x["base"] for x in st]
            pt, d = hier_boot(dsens, n_boot=NB)
            out["B_bouncer_steer"] = dict(point=float(pt), ci=list(ci(d)), frac_correct=float(np.mean(np.concatenate(dsens) > 0)))
    out["H4c"] = dict(p=float(max(pp)), desc="all identifiable targets R2 > 0.5 (intersection-union: max p)")
    pvals["H4c"] = max(pp)
    out["H4d"] = dict(p=float(max(dists)), desc="steering 1/m increases d v/d u on both systems (max p)")
    pvals["H4d"] = max(dists)

# ---------------- Experiment C
if os.path.exists("results/c/eval.pt"):
    R = torch.load("results/c/eval.pt", weights_only=False)
    tfn = [R[f"tf_seed{s}"]["norm"] for s in SEEDS]
    for h, other in (("H5a", [R[f"fixed_seed{s}"]["norm"] for s in SEEDS]), ("H5b", [R["ce"]["norm"] for s in SEEDS])):
        pt, d = hier_boot([o - t for o, t in zip(other, tfn)], n_boot=NB)
        rec(h, pt, d, p_greater(d), f"mean normalised cost: {'fixed' if h == 'H5a' else 'CE'} - TF > 0")
    pt, d = hier_boot([R[f"tf_seed{s}"]["info8"] - R["ce"]["info8"] for s in SEEDS], n_boot=NB)
    rec("H5d", pt, d, p_greater(d), "info gain at t=8: TF - CE > 0")
    summ = {}
    for k in [x for x in ["ce", "ce_dither", "etc", "ts"] if x in R] + [f"fixed_seed{s}" for s in SEEDS] + [f"tf_seed{s}" for s in SEEDS] + [f"bc_seed{s}" for s in SEEDS if f"bc_seed{s}" in R]:
        v = R[k]["norm"]
        summ[k] = dict(mean=float(v.mean()), median=float(np.median(v)), fail=float((v > 10).mean()),
                       info8=float(R[k]["info8"].mean()) if "info8" in R[k] else None)
    out["C_summary"] = summ
    for k in [x for x in ["ce_dither", "etc", "ts"] if x in R]:
        pt, d = hier_boot([R[k]["norm"] - t for t in tfn], n_boot=NB)
        out[f"H5c_{k}_minus_tf"] = dict(point=float(pt), ci=list(ci(d)), desc="exploratory two-sided, mean")
        pt, d = hier_boot([R[k]["norm"] - t for t in tfn], stat=np.median, n_boot=2000)
        out[f"H5c_{k}_minus_tf_median"] = dict(point=float(pt), ci=list(ci(d)), desc="median of paired differences")
        out[f"H5c_{k}_tf_wins"] = float(np.mean(np.concatenate([t < R[k]["norm"] for t in tfn])))
        pt, d = hier_boot([R[f"tf_seed{s}"]["info8"] - R[k]["info8"] for s in SEEDS], n_boot=NB)
        out[f"C_info8_tf_minus_{k}"] = dict(point=float(pt), ci=list(ci(d)))
    if all(f"bc_seed{s}" in R for s in SEEDS):
        pt, d = hier_boot([R[f"bc_seed{s}"]["norm"] - R[f"tf_seed{s}"]["norm"] for s in SEEDS], n_boot=NB)
        out["C_bc_minus_ft"] = dict(point=float(pt), ci=list(ci(d)), desc="imitation-only minus fine-tuned, mean normalised cost")
        pt, d = hier_boot([R[f"tf_seed{s}"]["info8"] - R[f"bc_seed{s}"]["info8"] for s in SEEDS], n_boot=NB)
        out["C_info8_ft_minus_bc"] = dict(point=float(pt), ci=list(ci(d)), desc="does cost fine-tuning add excitation beyond imitation?")
    out["C_tuning"] = {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in R["tuning"].items()}
    out["C_probe"] = {f"seed{s}": {f"{k[0]}@{k[1]}": round(v[0], 4) for k, v in R[f"tf_seed{s}"]["probe"].items()} for s in SEEDS}
    out["C_dev_ce"] = [R[f"tf_seed{s}"]["dev_ce"].mean(0).tolist() for s in SEEDS]
    out["C_steer_flip"] = [float(np.mean(np.sign(R[f"tf_seed{s}"]["steer_flip"]["u0"]) != np.sign(R[f"tf_seed{s}"]["steer_flip"]["u1"]))) for s in SEEDS]

# ---------------- Exploratory mechanism (A) and scale study
if os.path.exists("results/a/mech2.pt"):
    M2 = torch.load("results/a/mech2.pt", weights_only=False)
    out["A_mech2"] = {str(s): v for s, v in M2.items()}
if os.path.exists("results/a/mech3.pt"):
    M3 = torch.load("results/a/mech3.pt", weights_only=False)
    out["A_mech3"] = {str(s): v for s, v in M3.items()}
if os.path.exists("results/a/mech4.pt"):
    M4 = torch.load("results/a/mech4.pt", weights_only=False)
    out["A_mech4"] = {str(s): {"-".join(map(str, k)): v for k, v in r.items()} for s, r in M4.items()}
if os.path.exists("results/b/mech2.pt"):
    MB2 = torch.load("results/b/mech2.pt", weights_only=False)
    out["B_mech2"] = {"|".join(map(str, k)): v for k, v in MB2.items()}
if os.path.isdir("results/scale"):
    out["scale"] = {f[:-5]: {k: v for k, v in json.load(open(f"results/scale/{f}")).items() if k != "hist"}
                    for f in sorted(os.listdir("results/scale")) if f.endswith(".json")}

hol = holm(pvals)
out["holm"] = {k: dict(p=v[0], p_holm=v[1], reject=bool(v[2])) for k, v in hol.items()}
out["notes"] = notes
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/summary.json", "w"), indent=1, default=float)
lines = ["| Hypothesis | Estimate | 95% CI | p | p (Holm) | Supported |", "|---|---|---|---|---|---|"]
for k in sorted(hol):
    o = out[k]
    est = o.get("point", float("nan"))
    c = o.get("ci", [float("nan")] * 2)
    lines.append(f"| {k}: {o.get('desc', '')} | {est:.4g} | [{c[0]:.4g}, {c[1]:.4g}] | {hol[k][0]:.4g} | {hol[k][1]:.4g} | {'yes' if hol[k][2] else 'no'} |")
if "H2c" in out:
    lines.append(f"| H2c: {out['H2c']['desc']} | l_mom={out['H2c']['l_mom']}, l*={out['H2c']['l_star']} | - | - | - | {'yes' if out['H2c']['holds_all_seeds'] else 'no'} |")
open("results/SUMMARY.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
