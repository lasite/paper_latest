"""
fig4_data.py — Nonlinear regime classification on a 2D parameter grid.

Where fig 1(d,e) shows where *linear stability* of the homogeneous steady
state is lost (Hopf onset), fig 4 zooms inside the Hopf-unstable region
and asks what *kind* of nonlinear cycle the slab settles into. Each grid
point runs a full 1D PDE simulation; the long-time slab behaviour is
classified into one of:

  0  steady_cold         — no oscillation, uniformly swollen (φ < 0.5 everywhere)
  4  steady_collapsed    — no oscillation, uniformly collapsed (φ > 0.5 everywhere)
  5  steady_front        — no oscillation but a *frozen* partial-collapse
                            front: outer shell collapsed, core swollen
  1  bulk_hopf           — periodic, max φ < 0.5 everywhere (no LCST cross)
  2  lcst_front          — periodic, LCST is crossed in part of the slab
                            (the propagating-front regime of fig 3)
  3  global_collapse     — periodic, LCST crossed throughout the slab
 -1  failed              — solver failure or no usable analysis window

Default scan: (Bi_T, S_chi) — Bi_T tunes the Hopf onset; S_chi tunes the
LCST sensitivity. The two together cleanly separate the four regimes.
The working point of figs 2–3 lives inside the lcst_front region.

Each point is cheap (N=31, t_end=180); the full 10×10 grid finishes in
~10 min and is cached to data/fig4/fig4_grid_<axes>.npz.
"""
import os
# Each worker process runs scipy's BDF solver, which uses BLAS internally.
# When we fan out 24 worker processes, we must pin each one to a single
# thread or the BLAS pool will oversubscribe the CPU and grind. These
# must be set *before* numpy/scipy are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scan_optimized import Params, simulate
from fig2_data import WORKING_POINT
from fig3_data import cold_J_eq

DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "fig4"))


# Default number of parallel workers for grid scans. 24 is chosen for
# the user's workstation; override via the n_workers argument or the
# FIG4_WORKERS environment variable.
N_WORKERS_DEFAULT = int(os.environ.get("FIG4_WORKERS", 24))


# ── Simulation overrides for grid scans ───────────────────────────────
# Matches the fig 2 / 3 working-point spatial resolution (N=301, ℓ=0.01
# ⇒ dx/ℓ ≈ 0.33). Lighter only in the integration window: t_end=200
# vs 250 (still ≳ 8 oscillation periods inside T_ANA) and n_save=2000
# vs 10000 (figure-time density only). Each simulation is ~8 min on a
# single core with the sparse-LU path; the 15×15 grid finishes in
# ~3 hours on 12 workers.
SIM_OVERRIDES = dict(t_end=200.0, n_save=2000, N=301)
T_ANA = (120.0, 200.0)


# ── Default grid axes ─────────────────────────────────────────────────
# Main grid (Bi_T × S_chi): 25 × 25 — the regime-boundary panel needs
# this resolution to render boundaries as curves rather than staircases.
# Main grid (Bi_T × S_chi): 50 × 50 — bumped from 25 × 25 for APS
# print quality (smooth regime boundaries, denser SNIC sampling at the
# upper-S_chi exit).
# Secondary grid (Bi_T × Da): 50 × 30 — Da is auxiliary, fewer points
# OK but Bi_T axis shared with main for visual consistency.
BI_T_VALS  = np.geomspace(0.035, 0.40, 50)
S_CHI_VALS = np.linspace(0.0, 2.1, 50)
DA_VALS    = np.geomspace(0.5, 12.0, 30)


# ── Regime codes ──────────────────────────────────────────────────────
REG_FAILED            = -1
REG_STEADY_COLD       = 0
REG_BULK_HOPF         = 1
REG_LCST_FRONT        = 2
REG_GLOBAL_COLLAPSE   = 3
REG_STEADY_COLLAPSED  = 4
REG_STEADY_FRONT      = 5

REGIME_NAMES = {
    REG_FAILED:           "failed",
    REG_STEADY_COLD:      "steady_cold",
    REG_BULK_HOPF:        "bulk_hopf",
    REG_LCST_FRONT:       "lcst_front",
    REG_GLOBAL_COLLAPSE:  "global_collapse",
    REG_STEADY_COLLAPSED: "steady_collapsed",
    REG_STEADY_FRONT:     "steady_front",
}


# ── Classification thresholds ─────────────────────────────────────────
AMP_THRESH = 0.06       # min surface-J amplitude to count as oscillating
PHI_LCST   = 0.5        # LCST proxy — gel is "above LCST" when φ > 0.5
COLD_FRAC  = 0.60       # J_mean/J_eq above this → steady_cold else collapsed


def classify_point(p_dict, t_ana=T_ANA, amp_thresh=AMP_THRESH):
    """Run one simulation and return regime + diagnostic scalars.

    Diagnostics:
      J_amp_max      — max over ξ of (J_max − J_min) inside t_ana
      phi_max        — max over (ξ, t) of φ
      phi_max_min    — min over ξ of (max_t φ); >0.5 means *every* cell
                       crosses the LCST → global collapse oscillation
      J_mean         — mean over (ξ, t) of J inside t_ana
      J_eq           — cold-bath swelling reference for this parameter set
      surf_amp       — surface (ξ=1) amplitude of J in the analysis window
      period         — surface dominant period (NaN if non-oscillating)
    """
    p = Params(**{**p_dict, **SIM_OVERRIDES})
    out = dict(regime=REG_FAILED, J_amp_max=np.nan, phi_max=np.nan,
               phi_max_min=np.nan, J_mean=np.nan, J_eq=np.nan,
               surf_amp=np.nan, period=np.nan)
    try:
        d = simulate(p)
    except Exception as e:
        out["error"] = f"simulate-failed: {e}"
        return out

    if not d.get("success", True):
        out["error"] = "solver-not-success"
        return out

    t = d["t"]; J = d["J"]; theta = d["theta"]
    idx = (t >= t_ana[0]) & (t <= t_ana[1])
    if idx.sum() < 50:
        out["error"] = "short-window"
        return out

    Jw   = J[:, idx]
    th_w = theta[:, idx]
    phi  = p.phi_p0 / np.maximum(Jw, 1e-12)

    J_amp_per_xi = Jw.max(axis=1) - Jw.min(axis=1)
    phi_max_per_xi = phi.max(axis=1)

    out["J_amp_max"]   = float(J_amp_per_xi.max())
    out["phi_max"]     = float(phi_max_per_xi.max())
    out["phi_max_min"] = float(phi_max_per_xi.min())
    out["J_mean"]      = float(Jw.mean())
    out["surf_amp"]    = float(Jw[-1].max() - Jw[-1].min())

    # Period from surface time series (lightweight peak-finding)
    try:
        from scipy.signal import find_peaks
        Js = Jw[-1] - Jw[-1].mean()
        prom = max(0.05 * out["surf_amp"], 0.01)
        pk, _ = find_peaks(Js, prominence=prom, distance=3)
        if len(pk) >= 2:
            tt = t[idx]
            out["period"] = float(np.median(np.diff(tt[pk])))
    except Exception:
        pass

    J_eq = cold_J_eq(p_dict)
    if not np.isfinite(J_eq):
        J_eq = 1.0
    out["J_eq"] = float(J_eq)

    # ── Regime classification ─────────────────────────────────────────
    is_steady = (out["surf_amp"] < amp_thresh and
                 out["J_amp_max"] < amp_thresh)
    if is_steady:
        if out["phi_max"] < PHI_LCST:
            out["regime"] = REG_STEADY_COLD
        elif out["phi_max_min"] > PHI_LCST:
            out["regime"] = REG_STEADY_COLLAPSED
        else:
            # frozen partial-collapse front (outer shell collapsed,
            # core swollen, but no time-dependent dynamics)
            out["regime"] = REG_STEADY_FRONT
    else:
        if out["phi_max"] < PHI_LCST:
            # Below-LCST oscillation: in this LCST-coupled model these
            # are heavily damped transients toward steady_cold, not a
            # sustained limit cycle (the chi_1·phi term entrains any
            # thermally-driven swelling to LCST collapse). Lump into
            # steady_cold rather than promote a phantom regime.
            out["regime"] = REG_STEADY_COLD
        elif out["phi_max_min"] > PHI_LCST:
            out["regime"] = REG_GLOBAL_COLLAPSE
        else:
            out["regime"] = REG_LCST_FRONT

    return out


def _grid_path(param_x, param_y):
    return os.path.join(DATA_DIR, f"fig4_grid_{param_x}_{param_y}.npz")


def representative_runs(points, force=False):
    """Run + cache surface time series at the requested parameter points.

    ``points`` is a list of (label, p_dict) tuples, e.g.::

        [("steady_cold",   dict(WORKING_POINT, S_chi=0.2)),
         ("lcst_front_WP", dict(WORKING_POINT)),
         ("steady_front",  dict(WORKING_POINT, S_chi=2.0)),
         ("strong_lcst",   dict(WORKING_POINT, Bi_T=0.06, S_chi=1.2))]

    Returns a dict ``{label: {t, J_surf, J, theta, x, regime, ...}}``.
    Cached to data/fig4/fig4_repr.npz so the (slow) simulations only run
    once.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "fig4_repr.npz")
    out = {}
    if os.path.exists(path) and not force:
        try:
            z = np.load(path, allow_pickle=True)
            stored = {str(k) for k in z.files}
            for lbl, _ in points:
                if f"{lbl}__t" in stored:
                    out[lbl] = dict(
                        t       = z[f"{lbl}__t"],
                        J_surf  = z[f"{lbl}__J_surf"],
                        x       = z[f"{lbl}__x"],
                        J       = z[f"{lbl}__J"],
                        theta   = z[f"{lbl}__theta"],
                        regime  = int(z[f"{lbl}__regime"]),
                        params  = z[f"{lbl}__params"].item()
                                  if z[f"{lbl}__params"].dtype == object else {},
                    )
            if len(out) == len(points):
                return out
        except Exception:
            out = {}

    out = {}
    for lbl, p_dict in points:
        p = Params(**{**p_dict, **SIM_OVERRIDES})
        d = simulate(p)
        # Inline lightweight regime classification from this exact run
        # (avoids a second simulate() call inside classify_point).
        t = d["t"]; J = d["J"]
        idx = (t >= T_ANA[0]) & (t <= T_ANA[1])
        if idx.sum() < 50:
            regime = REG_FAILED
        else:
            Jw = J[:, idx]
            phi = p.phi_p0 / np.maximum(Jw, 1e-12)
            J_amp_max   = float((Jw.max(axis=1) - Jw.min(axis=1)).max())
            phi_max     = float(phi.max(axis=1).max())
            phi_max_min = float(phi.max(axis=1).min())
            surf_amp    = float(Jw[-1].max() - Jw[-1].min())
            is_steady = (surf_amp < AMP_THRESH and J_amp_max < AMP_THRESH)
            if is_steady:
                if phi_max < PHI_LCST:                     regime = REG_STEADY_COLD
                elif phi_max_min > PHI_LCST:               regime = REG_STEADY_COLLAPSED
                else:                                       regime = REG_STEADY_FRONT
            else:
                if phi_max < PHI_LCST:                     regime = REG_STEADY_COLD
                elif phi_max_min > PHI_LCST:               regime = REG_GLOBAL_COLLAPSE
                else:                                       regime = REG_LCST_FRONT
        out[lbl] = dict(
            t      = d["t"],
            J_surf = d["J"][-1],
            x      = d["x"],
            J      = d["J"],
            theta  = d["theta"],
            regime = int(regime),
            params = p_dict,
        )

    save_dict = {}
    for lbl, r in out.items():
        save_dict[f"{lbl}__t"]      = r["t"]
        save_dict[f"{lbl}__J_surf"] = r["J_surf"]
        save_dict[f"{lbl}__x"]      = r["x"]
        save_dict[f"{lbl}__J"]      = r["J"]
        save_dict[f"{lbl}__theta"]  = r["theta"]
        save_dict[f"{lbl}__regime"] = np.array(r["regime"])
        save_dict[f"{lbl}__params"] = np.array(r["params"], dtype=object)
    np.savez_compressed(path, **save_dict)
    return out


def _worker_classify(task):
    """Picklable worker entry point for ProcessPoolExecutor."""
    j, i, p_dict = task
    r = classify_point(p_dict)
    return j, i, r


def build_grid(param_x="Bi_T", x_vals=BI_T_VALS,
               param_y="S_chi", y_vals=S_CHI_VALS,
               base_overrides=None, force=False, verbose=True,
               n_workers=None):
    """Run the regime classification on every (x_vals × y_vals) point.

    Caches to data/fig4/fig4_grid_<x>_<y>.npz. Returns a dict with
    ``x``, ``y`` axes plus 2D arrays (rows = y_vals, cols = x_vals).

    Parallelism: uses ProcessPoolExecutor with ``n_workers`` processes
    (default 24, override via FIG4_WORKERS env var or argument). Each
    simulate() call is fully independent so the scan is embarrassingly
    parallel. Set ``n_workers=1`` to force serial execution.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _grid_path(param_x, param_y)
    if os.path.exists(path) and not force:
        z = np.load(path, allow_pickle=False)
        if verbose:
            print(f"  Using cache: {path}")
        return {k: z[k] for k in z.files} | dict(param_x=param_x,
                                                 param_y=param_y)

    NX, NY = len(x_vals), len(y_vals)
    regime      = np.full((NY, NX), REG_FAILED, dtype=int)
    J_amp_max   = np.full((NY, NX), np.nan)
    surf_amp    = np.full((NY, NX), np.nan)
    phi_max     = np.full((NY, NX), np.nan)
    phi_max_min = np.full((NY, NX), np.nan)
    J_mean      = np.full((NY, NX), np.nan)
    J_eq        = np.full((NY, NX), np.nan)
    period      = np.full((NY, NX), np.nan)

    base = dict(WORKING_POINT)
    if base_overrides:
        base.update(base_overrides)

    tasks = []
    for j, yv in enumerate(y_vals):
        for i, xv in enumerate(x_vals):
            p = dict(base)
            p[param_x] = float(xv)
            p[param_y] = float(yv)
            tasks.append((j, i, p))
    n_total = len(tasks)

    if n_workers is None:
        n_workers = N_WORKERS_DEFAULT
    n_workers = max(1, min(int(n_workers), n_total))

    if verbose:
        print(f"  Grid: {param_x} × {param_y} = {NX} × {NY} = {n_total} points")
        print(f"  Workers: {n_workers}")

    def _store(j, i, r):
        regime[j, i]      = r["regime"]
        J_amp_max[j, i]   = r["J_amp_max"]
        surf_amp[j, i]    = r["surf_amp"]
        phi_max[j, i]     = r["phi_max"]
        phi_max_min[j, i] = r["phi_max_min"]
        J_mean[j, i]      = r["J_mean"]
        J_eq[j, i]        = r["J_eq"]
        period[j, i]      = r["period"]

    if n_workers == 1:
        for k, (j, i, p) in enumerate(tasks, 1):
            r = classify_point(p)
            _store(j, i, r)
            if verbose:
                xv = p[param_x]; yv = p[param_y]
                print(f"  [{k:>3}/{n_total}] {param_x}={xv:6.3f} "
                      f"{param_y}={yv:6.3f}  "
                      f"regime={REGIME_NAMES.get(r['regime'],'?'):>16}  "
                      f"J_amp={r['J_amp_max']:.3f}  "
                      f"phi_max={r['phi_max']:.3f}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker_classify, t): t for t in tasks}
            done = 0
            for f in as_completed(futs):
                j, i, r = f.result()
                _store(j, i, r)
                done += 1
                if verbose:
                    xv = x_vals[i]; yv = y_vals[j]
                    print(f"  [{done:>3}/{n_total}] {param_x}={xv:6.3f} "
                          f"{param_y}={yv:6.3f}  "
                          f"regime={REGIME_NAMES.get(r['regime'],'?'):>16}  "
                          f"J_amp={r['J_amp_max']:.3f}  "
                          f"phi_max={r['phi_max']:.3f}")

    np.savez_compressed(path, x=x_vals, y=y_vals,
                        regime=regime,
                        J_amp_max=J_amp_max, surf_amp=surf_amp,
                        phi_max=phi_max, phi_max_min=phi_max_min,
                        J_mean=J_mean, J_eq=J_eq, period=period)
    if verbose:
        print(f"  Saved: {path}")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files} | dict(param_x=param_x,
                                             param_y=param_y)


def main_smoke():
    """Tiny 3×3 grid to validate the classifier before the full scan."""
    bi_t = np.array([0.06, 0.10, 0.20])
    s_chi = np.array([0.2, 1.0, 1.8])
    print("=== fig4 smoke test (3×3) ===")
    g = build_grid(param_x="Bi_T", x_vals=bi_t,
                   param_y="S_chi", y_vals=s_chi, force=True)
    print("\nregime grid (rows = S_chi ↑, cols = Bi_T →):")
    print(g["regime"])


def _summarize(name, g):
    print(f"\n--- {name} ---")
    print("regime grid (rows = y ↑, cols = x →):")
    print(g["regime"])
    counts = {}
    for code, n in REGIME_NAMES.items():
        c = int((g["regime"] == code).sum())
        if c:
            counts[n] = c
    print("counts:", counts)


def main_full():
    """Full (Bi_T, S_chi) grid on default axes."""
    print(f"=== fig4 main grid Bi_T × S_chi "
          f"({len(BI_T_VALS)}×{len(S_CHI_VALS)}) ===")
    g = build_grid(force=True)
    _summarize("Bi_T × S_chi", g)


def main_da():
    """Secondary (Bi_T, Da) grid at S_chi = WP."""
    print(f"=== fig4 cross-section grid Bi_T × Da "
          f"({len(BI_T_VALS)}×{len(DA_VALS)}) ===")
    g = build_grid(param_x="Bi_T", x_vals=BI_T_VALS,
                   param_y="Da",  y_vals=DA_VALS,
                   force=True)
    _summarize("Bi_T × Da", g)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run the 3×3 smoke test only")
    ap.add_argument("--da", action="store_true",
                    help="run the (Bi_T, Da) cross-section grid")
    ap.add_argument("--all", action="store_true",
                    help="run both the main grid and the Da grid")
    args = ap.parse_args()
    if args.smoke:
        main_smoke()
    elif args.da:
        main_da()
    elif args.all:
        main_full()
        main_da()
    else:
        main_full()
