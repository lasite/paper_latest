#!/usr/bin/env python3
"""
make_fig_all_regimes.py - Spatiotemporal mosaic of every PDE regime
the model supports, rendered as a 9-subpanel composite.

Layout (3x3 grid, panel labels (a)-(i), no row titles):

  (a) J(xi,t)         (b) theta(xi,t)         (c) <J(xi)>_t + envelope
  (d) J(xi,t)         (e) theta(xi,t)         (f) <J(xi)>_t + envelope
  (g) J(xi,t)         (h) theta(xi,t)         (i) <J(xi)>_t + envelope

The right column shows the time-mean J(xi) with a min/max envelope
band; the LCST threshold J = phi_p0/0.5 is marked as a grey dotted
horizontal line so that LCST-crossing regions (below the line) are
immediately visible -- in particular the thin collapsed band at
xi ~ 0.91 in the frozen-front row.

Row 1 (a-c): steady cold      (S_chi=0.20, default Bi_T)
Row 2 (d-f): LCST-front cycle (working point)
Row 3 (g-i): frozen front     (Bi_T=0.18, S_chi=1.50)

The hot-runaway hidden attractor is reached only from non-default
ICs and is shown in fig6 (manifold portrait), not here.

Two regimes allowed by the classifier (global_collapse and
steady_collapsed) are absent from any of our sims because of the
phi-dependent diffusion barrier that shields the core from collapse.
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT
from scan_optimized import (
    Params, finalize_params, rhs_mol_logJ, make_jac_sparsity,
    make_sparse_fd_jac, cell_centers, _LOG_J_MAX,
)
from scipy.integrate import solve_ivp

set_style()

DATA_DIR = _HERE.parent / "data" / "fig4"

LBL_FS  = 9
TICK_FS = 7
LEG_FS  = 7
CB_TICK_FS = 7
CB_LBL_FS  = 8


def simulate_one(p_dict, label, t_end=200.0, N=121, n_save=2000):
    """Run one PDE simulation with the default cold IC, return (t, x, J, theta)."""
    p_dict = {**p_dict, "N": N, "t_end": t_end, "n_save": n_save}
    p = Params(**p_dict)
    p = finalize_params(p)
    n3 = 3 * p.N

    x = cell_centers(p.N)
    log_J_min = np.log(p.phi_p0 * 1.02)
    J0 = np.maximum(p.J_init + p.eps_J * np.cos(np.pi * x),
                    np.exp(log_J_min) + 1e-6)
    u0 = np.maximum(p.u_init + p.eps_u * np.cos(np.pi * x), p.u_floor)
    t0 = p.theta_init + p.eps_theta * x
    y0 = np.concatenate([np.log(J0), J0 * u0, t0])

    rhs_fn = lambda t, y: rhs_mol_logJ(t, y, p)
    S = make_jac_sparsity(p.N)
    jac_sparse, _ = make_sparse_fd_jac(rhs_fn, S, n3)

    t0_wall = time.perf_counter()
    sol = solve_ivp(
        fun=rhs_fn, jac=jac_sparse,
        t_span=(0.0, p.t_end), y0=y0,
        t_eval=np.linspace(0, p.t_end, p.n_save),
        method=p.method, rtol=p.rtol, atol=p.atol, max_step=p.max_step,
    )
    print(f"  [{label}] {time.perf_counter()-t0_wall:.0f}s, "
          f"success={sol.success}, nfev={sol.nfev}")
    if not sol.success:
        raise RuntimeError(f"[{label}] {sol.message}")

    n = p.N
    J = np.exp(np.clip(sol.y[:n], log_J_min, _LOG_J_MAX))
    theta = sol.y[2*n:]
    return dict(t=sol.t, x=x, J=J, theta=theta)


def load_repr(name):
    """Load one regime trace from data/fig4/fig4_repr.npz."""
    z = np.load(DATA_DIR / "fig4_repr.npz", allow_pickle=True)
    return dict(t=z[f"{name}__t"], x=z[f"{name}__x"],
                J=z[f"{name}__J"], theta=z[f"{name}__theta"])


def gather_all_regimes():
    """Gather one (t, x, J, theta) snapshot per realised default-IC attractor."""
    return {
        "steady_cold":  dict(data=load_repr("steady_cold"),
                             params=dict(WORKING_POINT, S_chi=0.20)),
        "lcst_front":   dict(data=load_repr("lcst_front_WP"),
                             params=dict(WORKING_POINT)),
        "steady_front": dict(data=load_repr("steady_front"),
                             params=dict(WORKING_POINT, Bi_T=0.18,
                                         S_chi=1.50)),
    }


# ── Plot helpers ──────────────────────────────────────────────────────
def _heatmap(ax, t, x, Z, label, cmap="viridis", norm=None, fig=None):
    pcm = ax.pcolormesh(t, x, Z, cmap=cmap, norm=norm,
                        shading="auto", rasterized=True)
    ax.set_xlabel(r"$t$", fontsize=LBL_FS)
    ax.set_ylabel(r"$\xi$", fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    if fig is not None:
        cb = fig.colorbar(pcm, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=CB_TICK_FS)
        cb.set_label(label, fontsize=CB_LBL_FS)
    return pcm


def _window(d, t_window):
    t = d["t"]; idx = (t >= t_window[0]) & (t <= t_window[1])
    return dict(t=t[idx], x=d["x"], J=d["J"][:, idx],
                theta=d["theta"][:, idx])


def draw_J_heatmap(ax, run, t_window, fig):
    w = _window(run["data"], t_window)
    Jmin, Jmax = max(0.05, w["J"].min()), w["J"].max()
    norm_J = LogNorm(vmin=Jmin, vmax=Jmax)
    _heatmap(ax, w["t"], w["x"], w["J"], r"$J$",
             cmap="viridis", norm=norm_J, fig=fig)


def draw_theta_heatmap(ax, run, t_window, fig):
    w = _window(run["data"], t_window)
    th_max = max(0.5, w["theta"].max())
    norm_th = Normalize(vmin=0.0, vmax=th_max)
    _heatmap(ax, w["t"], w["x"], w["theta"], r"$\theta$",
             cmap="magma", norm=norm_th, fig=fig)


def draw_J_profile(ax, run, t_window):
    """Time-mean spatial profile J(xi) with min/max envelope.

    The mean is taken over the *second half* of t_window so that
    IC-relaxation transients (which are visible in the kymographs of
    the same row) do not bias the mean. For the steady regimes the
    envelope is invisibly thin; for the LCST-front cycle it shows the
    spatial extent of the oscillation (wide near the surface,
    vanishing in the cold core). The grey dotted line marks the LCST
    threshold J = phi_p0/0.5; any region below it has crossed LCST."""
    t_lo, t_hi = t_window
    profile_window = (0.5 * (t_lo + t_hi), t_hi)
    w = _window(run["data"], profile_window)
    x = w["x"]; J = w["J"]
    J_mean = J.mean(axis=1)
    J_min  = J.min(axis=1)
    J_max  = J.max(axis=1)

    if (J_max - J_min).max() > 0.05:
        ax.fill_between(x, J_min, J_max, color="#888888", alpha=0.28,
                        lw=0, label=r"min/max envelope")
    ax.plot(x, J_mean, color="#1f4e79", lw=1.1,
            label=r"$\langle J(\xi)\rangle$")
    ax.axhline(WORKING_POINT["phi_p0"] / 0.5, color="grey", lw=0.6,
               ls=":", label=r"$\varphi=0.5$ (LCST)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\xi$", fontsize=LBL_FS)
    ax.set_ylabel(r"$J$", fontsize=LBL_FS)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=TICK_FS, direction="out", length=2.2)
    ax.legend(fontsize=LEG_FS, loc="best", framealpha=0.85,
              handlelength=1.4, borderpad=0.35)
    ax.grid(alpha=0.3, lw=0.4)


def main():
    runs = gather_all_regimes()

    # All three rows use a uniform t in [0, 80] kymograph window so
    # the IC-to-attractor evolution is visible: cold-swollen heating
    # from theta=0 to theta~7, the LCST-front cycle developing from
    # cold IC into stable cycles, and the frozen-front collapsed band
    # forming and locking around t~20. The right-column time-mean
    # profile internally averages only over the second half of this
    # window (see draw_J_profile) so IC-relaxation transients do not
    # bias the steady-attractor structure shown by <J(xi)>.
    windows = {
        "steady_cold":  (0.0, 80.0),
        "lcst_front":   (0.0, 80.0),
        "steady_front": (0.0, 80.0),
    }

    order = ["steady_cold", "lcst_front", "steady_front"]
    n_rows = len(order)

    # 9-panel composite: 3 rows x 3 cols. Each axis gets a panel
    # label (a)-(i). No row titles or parameter subtitles -- regime
    # info goes into the LaTeX caption.
    fig = plt.figure(figsize=(PRE_DOUBLE * 1.05, 1.95 * n_rows))
    gs = gridspec.GridSpec(
        n_rows, 3, figure=fig,
        hspace=0.55, wspace=0.55,
        left=0.07, right=0.97, top=0.96, bottom=0.07,
    )

    panel_idx = 0
    for ri, key in enumerate(order):
        run = runs[key]
        win = windows[key]

        ax_J  = fig.add_subplot(gs[ri, 0])
        ax_th = fig.add_subplot(gs[ri, 1])
        ax_t  = fig.add_subplot(gs[ri, 2])

        draw_J_heatmap(ax_J, run, win, fig=fig)
        draw_theta_heatmap(ax_th, run, win, fig=fig)
        draw_J_profile(ax_t, run, win)

        for ax in (ax_J, ax_th, ax_t):
            add_panel_label(ax, chr(ord("a") + panel_idx),
                            x=-0.20, y=1.04, fontsize=10)
            panel_idx += 1

    out_dir = _HERE.parent / "Figure" / "pub"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig3.pdf"
    png = out_dir / "fig3.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")
    print(f"  Saved: {png}")


if __name__ == "__main__":
    main()
