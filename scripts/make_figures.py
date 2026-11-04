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
    for name, tg, col, lab in [("pendulum", "g", COL["bayes"], "pendulum: g"), ("pendulum", "inv_m", COL["ols"], "pendulum: 1/m"),
                               ("block", "mu", COL["tf"], r"block: $\mu$"), ("block", "inv_m", COL["gd1"], "block: 1/m")]:
        cur = S.get(f"B_{name}_probe_by_t", {}).get(tg)
        if cur:
            ax[2].plot(np.mean(cur, 0), color=col, label=lab)
    ax[2].set_ylim(-0.1, 1.0); ax[2].set_xlabel("context length t"); ax[2].set_ylabel("probe $R^2$ (fixed probe)")
    ax[2].legend(frameon=False, fontsize=6); ax[2].set_title("(c) physical parameters in the residual", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_b.pdf"); plt.close(fig)

# ---- Figure 3: Experiment C
if os.path.exists("results/c/eval.pt"):
    R = torch.load("results/c/eval.pt", weights_only=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    names = [("fixed_seed0", "fixed"), ("ce", "CE"), ("etc", "explore-then-CE"), ("ce_dither", "CE+dither"), ("tf_seed0", "transformer")]
    data = [np.log10(np.clip(R[k]["norm"], 1e-3, 1e6)) for k, _ in names]
    data[-1] = np.log10(np.clip(np.concatenate([R[f"tf_seed{s}"]["norm"] for s in range(3)]), 1e-3, 1e6))
    data[0] = np.log10(np.clip(np.concatenate([R[f"fixed_seed{s}"]["norm"] for s in range(3)]), 1e-3, 1e6))
    ax[0].boxplot(data, showfliers=False, whis=(5, 95))
    ax[0].set_xticks(range(1, 6)); ax[0].set_xticklabels([n for _, n in names], rotation=30, fontsize=6)
    ax[0].set_ylabel(r"$\log_{10}(J/J_{\rm oracle})$"); ax[0].set_title("(a) closed-loop cost", fontsize=8)
    dev = np.mean([R[f"tf_seed{s}"]["dev_ce"].mean(0) for s in range(3)], 0)
    uab = np.mean([R[f"tf_seed{s}"]["u_abs"].mean(0) for s in range(3)], 0)
    ax[1].plot(dev, color=COL["tf"], label=r"$\|u_{TF}-u_{CE}(\mathrm{own\ history})\|^2$")
    ax[1].plot(uab, color=COL["bayes"], ls="--", label=r"$|u_{TF}|$")
    ax[1].set_yscale("log"); ax[1].set_xlabel("time step t"); ax[1].legend(frameon=False, fontsize=6)
    ax[1].set_title("(b) excitation beyond certainty equivalence", fontsize=8)
    info = [R["ce"]["info8"], R["ce_dither"]["info8"], R["etc"]["info8"], np.concatenate([R[f"tf_seed{s}"]["info8"] for s in range(3)]), R["oracle"]["info8"]]
    ax[2].boxplot(info, showfliers=False, whis=(5, 95))
    ax[2].set_xticks(range(1, 6)); ax[2].set_xticklabels(["CE", "CE+dither", "expl.-then-CE", "transformer", "oracle LQR"], rotation=30, fontsize=6)
    ax[2].set_ylabel(r"$\log\det(\Lambda_0+\sigma^{-2}S_8)$"); ax[2].set_title("(c) information after 8 steps", fontsize=8)
    fig.tight_layout(); fig.savefig("paper/figs/fig_c.pdf"); plt.close(fig)
print("figures written")
