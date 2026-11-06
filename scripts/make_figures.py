"""Figures for the paper from results/summary.json (+ raw C results)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

S = json.load(open("results/summary.json"))
os.makedirs("paper/figs", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
COL = {"tf": "#c0392b", "bayes": "#2c3e50", "ridge": "#7f8c8d", "ols": "#2980b9", "gd1": "#27ae60", "theory": "#8e44ad"}

# ---- Figure 1: Experiment A, excess risk vs context length + probes by layer + patching by layer
if "A_curves" in S:
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    c = S["A_curves"]; t = np.arange(48)
    for k, lab in [("tf", "transformer"), ("bayes", "Bayes (exact)"), ("ols", "OLS"), ("gd1", "1-step GD")]:
        ax[0].plot(t[1:], np.array(c[k])[1:], label=lab, color=COL[k], lw=1.4 if k == "tf" else 1, ls="-" if k in ("tf", "bayes") else "--")
    ax[0].plot(t[8:], 0.06 / t[8:], color=COL["theory"], ls=":", label=r"$np\sigma^2/t$ (Cor. 1)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-3, 3)
    ax[0].set_xlabel("context length t"); ax[0].set_ylabel("excess risk"); ax[0].legend(frameon=False, fontsize=6)
    ax[0].set_title("(a) in-context prediction", fontsize=8)
    if "A_mech" in S:
        r2 = S["A_mech"]["r2_layers"]
        for k, lab, col in [("theta_bayes", r"$\hat\theta_{\rm Bayes}$", COL["bayes"]), ("theta_true", r"$\theta_{\rm true}$", COL["ridge"]),
                            ("moments", r"$(S_t,C_t)/(t{+}1)$", COL["gd1"]), ("post_logtrace", r"$\log\mathrm{tr}\,P_t$", COL["theory"])]:
            a = np.array(r2[k]); m = a.mean(0)
            ax[1].plot(range(5), m, marker="o", ms=3, label=lab, color=col)
            ax[1].fill_between(range(5), a.min(0), a.max(0), color=col, alpha=0.15)
        ax[1].set_xlabel("residual stream layer"); ax[1].set_ylabel("held-out probe $R^2$"); ax[1].set_ylim(-0.05, 1.02)
        ax[1].legend(frameon=False, fontsize=6); ax[1].set_title("(b) linear probes", fontsize=8)
        tl, tc = np.array(S["A_mech"]["tau_last"]), np.array(S["A_mech"]["tau_ctx"])
        ax[2].plot(range(5), tl.mean(0), marker="o", ms=3, label="patch query token", color=COL["tf"])
        ax[2].plot(range(5), tc.mean(0), marker="s", ms=3, label="patch context tokens", color=COL["ols"])
        ax[2].axhline(S["H3b"]["point"], color=COL["bayes"], ls="--", lw=0.8)
        ax[2].text(0.1, S["H3b"]["point"] + 0.03, r"6-d $\hat\theta$-subspace only", fontsize=6)
        ax[2].set_xlabel("layer patched"); ax[2].set_ylabel(r"transfer $\tau$"); ax[2].legend(frameon=False, fontsize=6)
        ax[2].set_title("(c) activation patching", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_a.pdf"); plt.close(fig)

# ---- Figure 2: Experiment B
if "B_block_curves" in S:
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for j, name in enumerate(["pendulum", "block"]):
        c = S[f"B_{name}_curves"]
        tf = np.mean([c[f"tf_err_seed{s}"] for s in range(3)], 0)
        mlp = np.mean([c[f"mlp_err_seed{s}"] for s in range(3)], 0)
        t = np.arange(len(tf))
        ax[j].plot(t, tf, color=COL["tf"], label="transformer")
        ax[j].plot(t, c["feature_rls_err"], color=COL["ols"], ls="--", label="physics-feature RLS")
        ax[j].plot(t, mlp, color=COL["gd1"], ls="-.", label="no-context MLP")
        ax[j].plot(t, c["nominal_err"], color=COL["ridge"], ls=":", label="nominal-param sim")
        ax[j].plot(t, c["oracle_err"], color=COL["bayes"], lw=0.8, label="true-param sim")
        ax[j].set_yscale("log"); ax[j].set_xlabel("context length t"); ax[j].set_ylabel("normalised MSE")
        ax[j].set_title(f"({'ab'[j]}) {name}", fontsize=8)
    ax[0].legend(frameon=False, fontsize=5.5)
    bm = S.get("B_mech2", {})
    items = [("pendulum", "g", "P: g"), ("pendulum", "inv_m", "P: 1/m"), ("pendulum", "c_over_m", "P: c/m"),
             ("block", "mu", r"K: $\mu$"), ("block", "inv_m", "K: 1/m"), ("bouncer", "g", "B: g"), ("bouncer", "inv_m", "B: 1/m"), ("bouncer", "zeta", r"B: $\zeta$")]
    fwv, rsv, labs = [], [], []
    for name, tg, lab in items:
        ks = [k for k in bm if k.startswith(name + "|") and k.endswith("|" + tg)]
        if not ks:
            continue
        per_seed = {}
        for k in ks:
            s_ = k.split("|")[1]
            per_seed.setdefault(s_, []).append(bm[k])
        fwv.append(np.mean([max(v[0] for v in vs) for vs in per_seed.values()]))
        rsv.append(np.mean([max(v[1] for v in vs) for vs in per_seed.values()]))
        labs.append(lab)
    xb = np.arange(len(labs))
    ax[2].bar(xb - 0.2, fwv, 0.4, color=COL["tf"], label="K,V fast weights")
    ax[2].bar(xb + 0.2, rsv, 0.4, color=COL["ridge"], label="residual stream")
    ax[2].set_xticks(xb); ax[2].set_xticklabels(labs, rotation=45, fontsize=6); ax[2].set_ylim(0, 1)
    ax[2].set_ylabel(r"probe $R^2$ at $t{=}47$"); ax[2].legend(frameon=False, fontsize=6)
    ax[2].set_title("(c) where the physics is readable", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_b.pdf"); plt.close(fig)

# ---- Figure 3: Experiment C
if os.path.exists("results/c/eval.pt"):
    R = torch.load("results/c/eval.pt", weights_only=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    pooled = lambda pre: np.concatenate([R[f"{pre}_seed{s}"]["norm"] for s in range(3)])
    names = [("fixed", pooled("fixed")), ("CE", R["ce"]["norm"]), ("expl.-then-CE", R["etc"]["norm"]),
             ("CE+dither", R["ce_dither"]["norm"]), ("Thompson", R["ts"]["norm"]),
             ("TF imitation", pooled("bc")), ("TF + cost FT", pooled("tf"))]
    data = [np.log10(np.clip(v, 1e-3, 1e6)) for _, v in names]
    ax[0].boxplot(data, showfliers=False, whis=(5, 95))
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xticks(range(1, len(names) + 1)); ax[0].set_xticklabels([n for n, _ in names], rotation=40, fontsize=5.5)
    ax[0].set_ylabel(r"$\log_{10}(J/J_{\rm oracle})$"); ax[0].set_title("(a) closed-loop cost", fontsize=8)
    diag = S["C_fail_diag"]["by_controller"]
    names_b = [("ce_dither", "CE+dither"), ("ts", "Thompson"), ("etc", "expl.-then-CE"), ("bc", "TF imitation"), ("tf", "TF + cost FT")]

    def fr(k, which):
        if k in ("bc", "tf"):
            return np.mean([diag[f"{k}_seed{s}"][which] for s in range(3)])
        return diag[k][which]
    xb = np.arange(len(names_b))
    ax[1].bar(xb - 0.2, [100 * fr(k, "fail_stable") for k, _ in names_b], 0.4, color=COL["bayes"], label="open-loop stable plants")
    ax[1].bar(xb + 0.2, [100 * fr(k, "fail_unstable") for k, _ in names_b], 0.4, color=COL["tf"], label="open-loop unstable plants")
    ax[1].set_xticks(xb); ax[1].set_xticklabels([n for _, n in names_b], rotation=40, fontsize=5.5)
    ax[1].set_ylabel(r"failure rate (%), $J/J_{\rm oracle}>10$"); ax[1].legend(frameon=False, fontsize=6)
    ax[1].set_title("(b) where the failures are", fontsize=8)
    pooled_i = lambda pre: np.concatenate([R[f"{pre}_seed{s}"]["info8"] for s in range(3)])
    info = [R["ce"]["info8"], R["ce_dither"]["info8"], R["etc"]["info8"], R["ts"]["info8"], pooled_i("bc"), pooled_i("tf"), R["oracle"]["info8"]]
    ax[2].boxplot(info, showfliers=False, whis=(5, 95))
    ax[2].set_xticks(range(1, 8)); ax[2].set_xticklabels(["CE", "CE+dither", "expl.-then-CE", "Thompson", "TF imitation", "TF + cost FT", "oracle LQR"], rotation=40, fontsize=5.5)
    ax[2].set_ylabel(r"$\log\det(\Lambda_0+\sigma^{-2}S_8)$"); ax[2].set_title("(c) information after 8 steps", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_c.pdf"); plt.close(fig)
# ---- Figure 4: mechanism (exploratory): Jacobian = Bayes estimate; estimate lives in key-value memory
if "A_mech3" in S and "A_mech4" in S:
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    M3 = S["A_mech3"]; t = np.arange(48)
    jb = np.array([M3[s]["jac_r2_bayes_by_t"] for s in M3])
    ax[0].plot(t[4:], jb.mean(0)[4:], color=COL["tf"], label=r"$R^2(J_t,\ \hat\theta^{\rm Bayes}_t)$ (left)")
    et = np.array([M3[s]["jac_err_true_by_t"] for s in M3]).mean(0)
    bt = np.array([M3[s]["bayes_err_true_by_t"] for s in M3]).mean(0)
    ax0 = ax[0].twinx()
    ax0.plot(t[4:], et[4:], color=COL["ridge"], ls="--", lw=0.9, label=r"$\|J_t-\theta\|^2$")
    ax0.plot(t[4:], bt[4:], color=COL["bayes"], ls=":", lw=0.9, label=r"$\|\hat\theta^{\rm Bayes}_t-\theta\|^2$")
    ax0.set_yscale("log"); ax0.set_ylabel("squared error", fontsize=6); ax0.tick_params(labelsize=6)
    ax0.legend(frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.82))
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("context length t"); ax[0].set_ylabel(r"$R^2$")
    ax[0].legend(frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    ax[0].set_title(r"(a) Jacobian $J_t=\partial\hat y_t/\partial z_t$", fontsize=8)
    fw = np.array([M3[s]["fw_r2"] for s in M3]); rr = np.array([M3[s]["resid_r2"] for s in M3])
    xs = np.arange(1, fw.shape[1] + 1)
    ax[1].plot(xs, fw.mean(0), marker="o", ms=3, color=COL["tf"], label="key-value fast weights")
    ax[1].plot(xs, rr.mean(0), marker="s", ms=3, color=COL["ridge"], label="residual stream (query)")
    ax[1].set_ylim(-0.05, 1); ax[1].set_xlabel("block"); ax[1].set_ylabel(r"probe $R^2$ for $\hat\theta_{47}$")
    ax[1].legend(frameon=False, fontsize=6); ax[1].set_title("(b) where the estimate is readable", fontsize=8)
    M4 = S["A_mech4"]
    keys = [("kv-0", "1"), ("kv-1", "2"), ("kv-2", "3"), ("kv-3", "4"), ("kv-2-3", "3+4"), ("kv-1-2-3", "2-4"), ("k-2-3", "K 3+4"), ("v-2-3", "V 3+4")]
    vals = np.array([[M4[s][k] for k, _ in keys] for s in M4])
    ax[2].bar(range(len(keys)), vals.mean(0), yerr=[vals.mean(0) - vals.min(0), vals.max(0) - vals.mean(0)], color=COL["ols"], capsize=2)
    ax[2].set_xticks(range(len(keys))); ax[2].set_xticklabels([l for _, l in keys], rotation=40, fontsize=6)
    ax[2].set_ylabel(r"transfer $\tau$"); ax[2].set_ylim(0, 1.05); ax[2].set_xlabel("context K,V patched (blocks)")
    ax[2].set_title("(c) causal: patch only K,V", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_mech.pdf"); plt.close(fig)

# ---- Figure 5: scale study (exploratory)
if "scale" in S and S["scale"]:
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.1))
    for tag, r in S["scale"].items():
        ex_tf, ex_r = np.array(r["excess_tf"]), np.array(r["excess_ridge"])
        ax[0].plot(np.arange(1, 48), (ex_tf / ex_r)[1:], label=f"{tag} ({r['params'] / 1e6:.1f}M)", lw=1)
    ax[0].axhline(1, color="k", lw=0.6); ax[0].set_xscale("log"); ax[0].set_xlabel("context length t")
    ax[0].set_ylabel("excess / Bayes excess"); ax[0].legend(frameon=False, fontsize=5); ax[0].set_title("(a) approximation floor", fontsize=8)
    tags = list(S["scale"]); jr = [np.mean(list(S["scale"][k]["jac_r2"].values())) for k in tags]
    ax[1].bar(range(len(tags)), jr, color=COL["tf"]); ax[1].set_xticks(range(len(tags))); ax[1].set_xticklabels(tags, rotation=40, fontsize=6)
    ax[1].set_ylabel(r"$R^2$(Jacobian, ridge $\hat\theta$)"); ax[1].set_ylim(0, 1); ax[1].set_title("(b) Jacobian = estimate", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_scale.pdf"); plt.close(fig)
print("figures written")
