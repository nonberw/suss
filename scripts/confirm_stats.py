"""Pre-registration 2 (PREREGISTRATION_2.md, tag prereg2-94b6cf8): the 11 registered tests with Holm-Bonferroni.
Part A: one-sided episode-level bootstrap (10,000 resamples), probes fixed at their fitted weights.
Part B: one-sided paired t-test across the 10 training seeds.
Writes results/confirm/summary.json and results/confirm/SUMMARY.md."""
import json

import numpy as np
from scipy import stats

OUT = "results/confirm"
NB = 10_000
rng = np.random.default_rng(20261107)
MODELS = ["long60k", "deep8", "wide256"]


def boot_mean(x, thr, greater):
    m = np.concatenate([x[rng.integers(0, len(x), (1000, len(x)))].mean(1) for _ in range(NB // 1000)])
    p = (m <= thr).mean() if greater else (m >= thr).mean()
    return float(x.mean()), np.percentile(m, [2.5, 97.5]).tolist(), float(max(p, 1 / (NB + 1)))


def r2_batch(pred, y, idx):
    """Held-out R^2 (mean over targets) for each bootstrap resample idx (b, n)."""
    yy, pp = y[idx], pred[idx]
    sse = ((pp - yy) ** 2).sum(1)
    sst = ((yy - yy.mean(1, keepdims=True)) ** 2).sum(1)
    return (1 - sse / sst).mean(1)


def boot_gap(pk, pr, y, thr):
    gaps = []
    for _ in range(NB // 500):
        idx = rng.integers(0, len(y), (500, len(y)))
        gaps.append(r2_batch(pk, y, idx) - r2_batch(pr, y, idx))
    gaps = np.concatenate(gaps)
    full = np.arange(len(y))[None]
    point = float(r2_batch(pk, y, full)[0] - r2_batch(pr, y, full)[0])
    return point, np.percentile(gaps, [2.5, 97.5]).tolist(), float(max((gaps <= thr).mean(), 1 / (NB + 1)))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


tests = []
for m in MODELS:
    z = np.load(f"{OUT}/a_{m}.npz")
    est, ci, p = boot_mean(z["tau_late"], 0.8, greater=True)
    tests.append(dict(id=f"H6a|{m}", what="KV-only patching of the late blocks transfers tau >= 0.8", est=est, ci=ci, p=p))
    est, ci, p = boot_mean(z["tau_b1"], 0.3, greater=False)
    tests.append(dict(id=f"H6b|{m}", what="KV-only patching of block 1 transfers tau <= 0.3", est=est, ci=ci, p=p))
    est, ci, p = boot_gap(z["pred_kv"], z["pred_res"], z["y_test"], 0.15)
    tests.append(dict(id=f"H6c|{m}", what="fast-weight probe beats best residual readout by >= 0.15 R^2 (d_model dims)",
                      est=est, ci=ci, p=p))
b = json.load(open(f"{OUT}/b_summary.json"))
seeds = sorted(b["per_seed"], key=int)
dtr = np.array([b["per_seed"][s]["ridge_train"] for s in seeds])
for hid, key, what in (("H7a", "bayes_new", "Jacobian closer to training-prior ridge than to new-prior Bayes"),
                       ("H7b", "ols", "Jacobian closer to training-prior ridge than to OLS")):
    d = np.array([b["per_seed"][s][key] for s in seeds]) - dtr
    tt = stats.ttest_1samp(d, 0.0, alternative="greater")
    sem = d.std(ddof=1) / np.sqrt(len(d))
    tests.append(dict(id=hid, what=what, est=float(d.mean()), ci=[float(d.mean() - 2.262 * sem), float(d.mean() + 2.262 * sem)],
                      p=float(tt.pvalue), n_pos=int((d > 0).sum()), n=len(d)))
adj = holm(np.array([t["p"] for t in tests]))
for t, a in zip(tests, adj):
    t["p_holm"] = float(a)
    t["supported"] = bool(a < 0.05)
verdict = {h: all(t["supported"] for t in tests if t["id"].startswith(h + "|")) for h in ("H6a", "H6b", "H6c")}
verdict.update({t["id"]: t["supported"] for t in tests if t["id"].startswith("H7")})
json.dump({"tests": tests, "verdict": verdict, "a": json.load(open(f"{OUT}/a_summary.json")),
           "b_per_seed": b["per_seed"]}, open(f"{OUT}/summary.json", "w"), indent=1)
lines = ["# Pre-registration 2: results", "", "Registered in `PREREGISTRATION_2.md` (tag `prereg2-94b6cf8`) before these analyses ran.", "",
         "| Test | Estimate | 95% CI | p | p (Holm, 11 tests) | Supported |", "|---|---|---|---|---|---|"]
for t in tests:
    extra = f" ({t['n_pos']}/{t['n']} seeds)" if "n_pos" in t else ""
    lines.append(f"| {t['id']}: {t['what']} | {t['est']:.3f}{extra} | [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}] | {t['p']:.4g} | "
                 f"{t['p_holm']:.4g} | {'yes' if t['supported'] else 'no'} |")
lines += ["", "Verdicts (a hypothesis holds for the KV account only if it holds in every model): " +
          ", ".join(f"{k} {'supported' if v else 'not supported'}" for k, v in verdict.items())]
open(f"{OUT}/SUMMARY.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
