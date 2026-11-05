"""MuJoCo systems with randomized physics (Experiment B) and dataset generation."""
from __future__ import annotations

import numpy as np
import mujoco

DT_PHYS, SKIP = 0.01, 5
DT = DT_PHYS * SKIP
T = 48

PENDULUM_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <inertial pos="0 0 -0.5" mass="1" diaginertia="1e-5 1e-5 1e-5"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.01" mass="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" gear="1" ctrllimited="false"/></actuator>
</mujoco>"""

BLOCK_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" friction="0.5 0.005 0.0001"/>
    <body name="box" pos="0 0 0.1">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <geom name="boxg" type="box" size="0.1 0.1 0.1" mass="1" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator><motor joint="x" gear="1" ctrllimited="false"/></actuator>
</mujoco>"""

BOUNCER_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" solref="0.02 0.3"/>
    <body name="ball" pos="0 0 1">
      <joint name="z" type="slide" axis="0 0 1"/>
      <geom name="ballg" type="sphere" size="0.1" mass="1" solref="0.02 0.3"/>
    </body>
  </worldbody>
  <actuator><motor joint="z" gear="1" ctrllimited="false"/></actuator>
</mujoco>"""

L_POLE = 0.5
SYSTEMS = {
    "pendulum": dict(xml=PENDULUM_XML, params=dict(m=(0.5, 2.0), c=(0.02, 0.3), g=(4.0, 16.0)), u_sd=2.0, u_clip=5.0, s_p=0.2),
    "block": dict(xml=BLOCK_XML, params=dict(m=(0.5, 3.0), mu=(0.1, 1.0)), u_lim=15.0, s_p=0.2),
    # EXPLORATORY third system (added post hoc, see DEVIATIONS.md): a ball bouncing on the floor, with
    # unknown mass, gravity and contact damping ratio (sets the restitution), pushed by a random vertical force.
    "bouncer": dict(xml=BOUNCER_XML, params=dict(m=(0.5, 2.0), g=(4.0, 16.0), zeta=(0.2, 1.0)), u_lim=None, s_p=0.2, skip=25),
}
NOMINAL = {"pendulum": dict(m=1.25, c=0.16, g=10.0), "block": dict(m=1.75, mu=0.55), "bouncer": dict(m=1.25, g=10.0, zeta=0.6)}


def build(name):
    return mujoco.MjModel.from_xml_string(SYSTEMS[name]["xml"])


def set_params(model, name, p):
    if name == "pendulum":
        model.body_mass[1] = p["m"]
        model.dof_damping[0] = p["c"]
        model.opt.gravity[2] = -p["g"]
    elif name == "block":
        model.body_mass[1] = p["m"]
        for gid in range(model.ngeom):
            model.geom_friction[gid, 0] = p["mu"]
    else:
        model.body_mass[1] = p["m"]
        model.opt.gravity[2] = -p["g"]
        for gid in range(model.ngeom):
            model.geom_solref[gid] = [0.02, p["zeta"]]


def observe(name, d):
    if name == "pendulum":
        q, w = d.qpos[0], d.qvel[0]
        return np.array([np.sin(q), np.cos(q), w])
    return np.array([d.qpos[0], d.qvel[0]])


def step(model, d, u, noise=None):
    d.ctrl[0] = u
    for k in range(len(noise) if noise is not None else round(DT / model.opt.timestep)):
        if noise is not None:
            d.qfrc_applied[0] = noise[k]
        mujoco.mj_step(model, d)
    d.qfrc_applied[:] = 0


def reset(name, model, d, rng):
    mujoco.mj_resetData(model, d)
    if name == "pendulum":
        d.qpos[0] = rng.uniform(-np.pi, np.pi)
        d.qvel[0] = rng.normal()
    elif name == "bouncer":
        d.qpos[0] = rng.uniform(0.5, 2.0) - 1.0  # joint offset from body pos z=1
        d.qvel[0] = rng.normal(0, 2.0)
    else:
        d.qpos[:] = 0
        for _ in range(30):  # settle onto the floor
            mujoco.mj_step(model, d)
        d.qpos[0] = 0.0
        d.qvel[:] = 0
    mujoco.mj_forward(model, d)


def sample_params(name, rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in SYSTEMS[name]["params"].items()}


def sample_u(name, rng, n):
    s = SYSTEMS[name]
    if name == "pendulum":
        return np.clip(rng.normal(0, s["u_sd"], n), -s["u_clip"], s["u_clip"])
    if name == "bouncer":  # small random push (|u| <= 1 N < min weight 2 N): the ball bounces under gravity
        return rng.uniform(-1.0, 1.0, n)
    return rng.uniform(-s["u_lim"], s["u_lim"], n)


def n_sub(name):
    return SYSTEMS[name].get("skip", SKIP)


def gen_episodes(name, n, seed):
    """Returns dict with obs (n,T+1,do), u (n,T), params (n,k), oracle (n,T,do), nominal (n,T,do)."""
    rng = np.random.default_rng(seed)
    m_true, m_nom = build(name), build(name)
    set_params(m_nom, name, NOMINAL[name])
    d, d2 = mujoco.MjData(m_true), mujoco.MjData(m_true)
    dn = mujoco.MjData(m_nom)
    keys = list(SYSTEMS[name]["params"])
    do = 3 if name == "pendulum" else 2
    obs = np.zeros((n, T + 1, do)); us = np.zeros((n, T)); prm = np.zeros((n, len(keys)))
    orc = np.zeros((n, T, do)); nom = np.zeros((n, T, do))
    s_p = SYSTEMS[name]["s_p"]
    for e in range(n):
        p = sample_params(name, rng)
        set_params(m_true, name, p)
        prm[e] = [p[k] for k in keys]
        reset(name, m_true, d, rng)
        u = sample_u(name, rng, T)
        us[e] = u
        for t in range(T):
            obs[e, t] = observe(name, d)
            for dd, mm, buf in ((d2, m_true, orc), (dn, m_nom, nom)):
                dd.qpos[:] = d.qpos; dd.qvel[:] = d.qvel; dd.time = d.time
                dd.qacc_warmstart[:] = d.qacc_warmstart
                mujoco.mj_forward(mm, dd)
                step(mm, dd, u[t])
                buf[e, t] = observe(name, dd)
            step(m_true, d, u[t], rng.normal(0, s_p, n_sub(name)))
        obs[e, T] = observe(name, d)
    return dict(obs=obs, u=us, params=prm, oracle=orc, nominal=nom, keys=np.array(keys))


def _worker(args):
    return gen_episodes(*args)


def gen_parallel(name, n, seed, workers=20, chunk=500):
    from multiprocessing import Pool
    jobs = [(name, min(chunk, n - i), seed * 100_000 + i // chunk) for i in range(0, n, chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_worker, jobs)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "keys"}
    out["keys"] = parts[0]["keys"]
    return out


# ----- physics-feature recursive least squares baseline -----

def feature_rls(name, obs, u, lam=1e-2):
    """Online ridge on physics features; prediction of obs_{t+1} from pairs s<t. obs (n,T+1,do)."""
    n = obs.shape[0]
    if name == "pendulum":
        q = np.arctan2(obs[..., 0], obs[..., 1]); w = obs[..., 2]
        phi = np.stack([u, w[:, :T], np.sin(q[:, :T]), np.ones_like(u)], -1)
        dv = w[:, 1:] - w[:, :T]
    elif name == "block":
        x, w = obs[..., 0], obs[..., 1]
        phi = np.stack([u, np.sign(w[:, :T]), w[:, :T], np.ones_like(u)], -1)
        dv = w[:, 1:] - w[:, :T]
    else:  # bouncer: ballistic flight + linear spring-damper contact model (contact indicator c)
        x, w = obs[..., 0], obs[..., 1]
        pen = (x[:, :T] + 1.0) - 0.1  # height of the ball's bottom above the floor
        c = (pen < 0.02).astype(float)
        phi = np.stack([u, np.ones_like(u), c, c * w[:, :T], c * pen], -1)
        dv = w[:, 1:] - w[:, :T]
    k = phi.shape[-1]
    S = np.cumsum(phi[..., :, None] * phi[..., None, :], 1) - phi[..., :, None] * phi[..., None, :]
    c = np.cumsum(phi * dv[..., None], 1) - phi * dv[..., None]
    beta = np.linalg.solve(S + lam * np.eye(k), c[..., None])[..., 0]
    dvh = (beta * phi).sum(-1)
    wn = w[:, :T] + dvh
    if name == "pendulum":
        qn = q[:, :T] + DT * 0.5 * (w[:, :T] + wn)
        return np.stack([np.sin(qn), np.cos(qn), wn], -1), beta
    xn = x[:, :T] + DT * 0.5 * (w[:, :T] + wn)
    return np.stack([xn, wn], -1), beta
