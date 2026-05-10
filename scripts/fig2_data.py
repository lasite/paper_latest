"""
fig2_data.py — Shared simulation + cache loader for the Fig 2 panel scripts.

Caches the α=0.03 working-point simulation in data/fig2/cache.npz.
Each per-panel script (make_fig2[a-f]_*.py) imports `load_cache()` from
this module so the simulation runs at most once across the whole figure.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "fig2"))
FIG_DIR  = os.path.normpath(os.path.join(_HERE, "..", "Figure", "fig2"))
CACHE    = os.path.join(DATA_DIR, "cache.npz")

# Time window used by all data-driven panels (b, c, d, e, f).
T_START = 180.0
T_END   = 240.0

# Working-point parameters (default Table I) — kept here as a single source
# of truth so the caption / docs stay in sync with the simulation.
WORKING_POINT = dict(
    Da=4.0, S_chi=1.0, Gamma_A=1.5,
    Bi_T=0.10, Bi_c=0.70, Bi_mu=1.0,
    D0=2.0, m_act=6.0, m_diff=2.0, m_mob=1.0,
    alpha=0.03, delta=0.08, ell=0.01,
    Omega_e=0.12, phi_p0=0.15, chi_inf=0.60, chi1=1.10, eps_T=0.03,
    N=301, t_end=250.0, n_save=10000,
    rtol=1e-6, atol=1e-8,
)


def load_cache():
    """Return the cached simulation result, running the simulation if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,  exist_ok=True)
    if os.path.exists(CACHE):
        print(f"  Using cache: {CACHE}")
        return np.load(CACHE)

    print("  Running simulation (α=0.03) ...")
    import sys
    sys.path.insert(0, _HERE)
    from scan_optimized import Params, simulate
    p = Params(**WORKING_POINT)
    result = simulate(p)
    t, J, u, theta = result["t"], result["J"], result["u"], result["theta"]
    phi = p.phi_p0 / J
    access = np.power(np.clip(1.0 - phi, 1e-12, 1.0), p.m_act)
    np.savez(CACHE, x=result["x"], t=t, J=J, u=u, theta=theta, access=access)
    print(f"  Saved cache: {CACHE}")
    return np.load(CACHE)


def time_window(t):
    """Return boolean mask selecting [T_START, T_END] from time array."""
    return (t >= T_START) & (t <= T_END)


def save_panel(fig, name):
    """Save fig as PDF + PNG to Figure/fig2/<name>.{pdf,png}."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}.pdf")
