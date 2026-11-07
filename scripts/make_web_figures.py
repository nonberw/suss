"""PNG versions of the paper figures for the README / project page, plus a hero image and a 1280x640 social preview."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf

os.makedirs("docs/figs", exist_ok=True)
for name in ["fig_a", "fig_mech", "fig_b", "fig_c"]:
    d = pymupdf.open(f"paper/figs/{name}.pdf")
    d[0].get_pixmap(dpi=220).save(f"docs/figs/{name}.png")

S = json.load(open("results/summary.json"))
COL = {"tf": "#c0392b", "res": "#7f8c8d", "kv": "#2980b9", "bayes": "#2c3e50"}


def hero(path, w, h, dpi, title_size, show_title=True):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    if show_title:
        fig.text(0.03, 0.90, "SUSS", fontsize=title_size, weight="bold", color="#1a1a1a")
        fig.text(0.03, 0.82, "What a transformer learns when it adapts to unknown physics in-context",
                 fontsize=title_size * 0.42, color="#333333")
    top = 0.70 if show_title else 0.85
    # (1) readability: residual stream vs key-value memory (dimension-matched), 10 seeds
    ax1 = fig.add_axes([0.06, 0.14, 0.26, top - 0.14])
    m5, m6 = S["A_mech5"], S["A_mech6"]
    # the best residual-stream readout (mean-pooled, 128-d) next to the dimension-matched key-value readout
    bars = [("resid.\nquery", m5["resid_q|theta_bayes"]["mean"], COL["res"]),
            ("resid.\nmean-pooled", float(np.mean(m6["resid_mean|theta_bayes"])), COL["res"]),
            ("K,V\n128-d", m5["fw_rp128|theta_bayes"]["mean"], COL["kv"]),
            ("K,V\nfull", m5["fw|theta_bayes"]["mean"], COL["kv"])]
    ax1.bar(range(4), [b[1] for b in bars], color=[b[2] for b in bars])
    ax1.set_xticks(range(4)); ax1.set_xticklabels([b[0] for b in bars], fontsize=7)
    ax1.set_ylim(0, 1); ax1.set_ylabel(r"probe $R^2$ for the Bayes estimate", fontsize=8)
    ax1.set_title("where the estimate is readable", fontsize=9)
    # (2) causal: patch only K,V of blocks
    ax2 = fig.add_axes([0.40, 0.14, 0.26, top - 0.14])
    M4 = S["A_mech4"]
    keys = [("kv-0", "block 1"), ("kv-2", "block 3"), ("kv-3", "block 4"), ("kv-2-3", "blocks 3+4")]
    vals = [np.mean([M4[s][k] for s in M4]) for k, _ in keys]
    ax2.bar(range(4), vals, color=[COL["res"], COL["kv"], COL["kv"], COL["tf"]])
    ax2.set_xticks(range(4)); ax2.set_xticklabels([l for _, l in keys], fontsize=7)
    ax2.set_ylim(0, 1); ax2.set_ylabel("behaviour transferred", fontsize=8)
    ax2.set_title("swap only context keys & values", fontsize=9)
    # (3) control: median vs failure
    ax3 = fig.add_axes([0.74, 0.14, 0.23, top - 0.14])
    cs = S["C_summary"]
    pts = [("CE+dither", cs["ce_dither"]), ("Thompson", cs["ts"]), ("explore\nthen CE", cs["etc"]),
           ("transformer", {"median": np.mean([cs[f"tf_seed{s}"]["median"] for s in range(3)]), "fail": np.mean([cs[f"tf_seed{s}"]["fail"] for s in range(3)])})]
    for name, d in pts:
        c = COL["tf"] if name == "transformer" else COL["bayes"]
        ax3.scatter(d["median"], 100 * d["fail"], color=c, s=40)
        ax3.annotate(name, (d["median"], 100 * d["fail"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax3.set_xlabel(r"median cost / oracle", fontsize=8); ax3.set_ylabel("plants failed (%)", fontsize=8)
    ax3.set_xlim(1.4, 2.1); ax3.set_ylim(0, 13)
    ax3.set_title("in-context controller", fontsize=9)
    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


hero("docs/figs/hero.png", 1600, 560, 200, 22, show_title=False)
hero("docs/figs/social_preview.png", 1280, 640, 160, 26, show_title=True)
print("web figures written:", sorted(os.listdir("docs/figs")))
