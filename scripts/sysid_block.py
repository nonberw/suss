"""EXPLORATORY baseline (added after external review): structured system identification with the exact
simulator for the pushed block. For each test episode, every candidate (m, mu) on a 16x16 grid is simulated one
step from the observed state (x_t, xdot_t) at every t; the causal predictor at time t uses the candidate with the
smallest accumulated one-step error on the pairs s < t (nominal parameters at t = 0). This is the strongest explicit
system-ID baseline available: it has the true model class and the true simulator, and only has to pick parameters.
Also reports oracle-from-obs: the true parameters, but the state reconstructed from the observation only (z settled),
which bounds the error this baseline pays for not seeing the full MuJoCo state. Writes results/b/block_sysid.npz."""
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import mj

NAME = "block"
GRID_M = np.linspace(0.5, 3.0, 16)
GRID_MU = np.linspace(0.1, 1.0, 16)


def one_step_preds(args):
    """For a chunk of episodes: per candidate, per t, the predicted next observation from (x_t, v_t)."""
    obs, u, params = args
    model = mj.build(NAME)
    d = mj.MjData = mujoco.MjData(model)
    n, T1, _ = obs.shape
    T = T1 - 1
    cands = [(m, mu) for m in GRID_M for mu in GRID_MU] + [tuple(params[e]) for e in range(n)]  # true params last (per episode)
    preds = np.zeros((n, len(GRID_M) * len(GRID_MU) + 1, T, 2))
    rng = np.random.default_rng(0)
    for ci, (m, mu) in enumerate(cands[:len(GRID_M) * len(GRID_MU)]):
        mj.set_params(model, NAME, dict(m=m, mu=mu))
        mj.reset(NAME, model, d, rng)  # settles z
        z_rest = d.qpos[1]
        for e in range(n):
            for t in range(T):
                d.qpos[0], d.qpos[1] = obs[e, t, 0], z_rest
                d.qvel[:] = 0.0
                d.qvel[0] = obs[e, t, 1]
                mujoco.mj_forward(model, d)
                mj.step(model, d, u[e, t])
                preds[e, ci, t] = mj.observe(NAME, d)
    for e in range(n):  # oracle-from-obs
        m, mu = params[e]
        mj.set_params(model, NAME, dict(m=m, mu=mu))
        mj.reset(NAME, model, d, rng)
        z_rest = d.qpos[1]
        for t in range(T):
            d.qpos[0], d.qpos[1] = obs[e, t, 0], z_rest
            d.qvel[:] = 0.0
            d.qvel[0] = obs[e, t, 1]
            mujoco.mj_forward(model, d)
            mj.step(model, d, u[e, t])
            preds[e, -1, t] = mj.observe(NAME, d)
    return preds


if __name__ == "__main__":
    D = np.load(f"data/{NAME}_test.npz")
    obs, u, params = D["obs"], D["u"], D["params"]
    E, T = u.shape
    chunk = 128
    jobs = [(obs[i:i + chunk], u[i:i + chunk], params[i:i + chunk]) for i in range(0, E, chunk)]
    with Pool(int(os.environ.get("SUSS_WORKERS", "6"))) as pool:
        parts = pool.map(one_step_preds, jobs)
    preds = np.concatenate(parts)  # (E, n_cand+1, T, 2)
    nxt = obs[:, 1:]
    var = nxt.reshape(-1, 2).var(0)
    err = ((preds[:, :-1] - nxt[:, None]) ** 2 / var).mean(-1)  # (E, n_cand, T) normalised per-step error
    cum = np.cumsum(err, axis=2) - err  # error on pairs s < t
    nominal = np.argmin((GRID_M[:, None] - mj.NOMINAL[NAME]["m"]) ** 2 + (GRID_MU[None, :] - mj.NOMINAL[NAME]["mu"]) ** 2)
    choice = np.argmin(cum, axis=1)  # (E, T)
    choice[:, 0] = nominal
    sysid_pred = np.take_along_axis(preds[:, :-1], choice[:, None, :, None], axis=1)[:, 0]  # (E, T, 2)
    e_sys = (sysid_pred - nxt) ** 2 / var
    e_orc = (preds[:, -1] - nxt) ** 2 / var
    chosen = np.stack([GRID_M[choice // len(GRID_MU)], GRID_MU[choice % len(GRID_MU)]], -1)  # (E,T,2) identified (m, mu)
    os.makedirs("results/b", exist_ok=True)
    np.savez(f"results/b/{NAME}_sysid.npz", sysid_err=e_sys.mean(-1), sysid_vel=e_sys[..., 1], oracle_obs_err=e_orc.mean(-1),
             oracle_obs_vel=e_orc[..., 1], chosen=chosen, params=params)
    sl = slice(24, 48)
    print("block structured sysID: err(t=24..47) %.5f  oracle-from-obs %.5f" % (e_sys.mean(-1)[:, sl].mean(), e_orc.mean(-1)[:, sl].mean()))
    print("identified mu corr with true at t=47: %.3f, m corr: %.3f" % (np.corrcoef(chosen[:, -1, 1], params[:, 1])[0, 1], np.corrcoef(chosen[:, -1, 0], params[:, 0])[0, 1]))
