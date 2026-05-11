#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_overswollen_diag.py — diagnostic for the "over-swollen SS" Phase C found
between Da_c^0D and Da_c^PDE.

At Bi_T = 0.10, Da around 0.15-0.23 the system relaxes to a state with
J_surf ~= 3.8-4.0 (mean), neither cold-swollen (J~=1.3) nor LF cycle.
This script:
  - Runs the PDE at (Bi_T=0.10, Da=0.20) with extended t_end = 800
  - Verifies the state is truly steady (J_surf time series flat at late t)
  - Captures the spatial profile (J, theta, phi vs xi) at t_end
  - Saves the full result for later inspection

Output
  data/iv_c/phaseC_diag/overswollen_BiT010_Da020.npz
  Figure/pub/iv_c_overswollen_diag.{pdf,png}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from scan_optimized import Params, simulate, finalize_params
from style_pub import set_style, PRE_DOUBLE   # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = _HERE.parent / "data" / "iv_c" / "phaseC_diag"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = _HERE.parent / "Figure" / "pub"


def main():
    Bi_T = 0.10
    Da = 0.20
    N = 301
    t_end = 800.0
    n_save = 4000

    print("=" * 70, flush=True)
    print(f" Over-swollen SS diagnostic at Bi_T={Bi_T}, Da={Da}", flush=True)
    print(f" N={N}, t_end={t_end}, n_save={n_save}", flush=True)
    print("=" * 70, flush=True)

    p = Params(Bi_T=Bi_T, Da=Da, N=N, t_end=t_end, n_save=n_save)
    p = finalize_params(p)

    t0 = time.perf_counter()
    r = simulate(p)
    wall = time.perf_counter() - t0
    print(f"\n  Simulation wall-clock: {wall:.1f}s ({wall/60:.2f} min)",
          flush=True)

    t = r["t"]
    x = r["x"]
    J = r["J"]
    theta = r["theta"]
    phi = r["phi"]

    J_surf = J[-1]                # (n_t,)
    theta_surf = theta[-1]
    J_core = J[0]                 # innermost cell
    theta_core = theta[0]

    # Steadiness diagnostics on the late window
    t_late = (t >= 600.0) & (t <= 800.0)
    J_surf_late = J_surf[t_late]
    theta_surf_late = theta_surf[t_late]
    print(f"\n  Late window (600 < t < 800), n={t_late.sum()}:", flush=True)
    print(f"    J_surf:       mean={J_surf_late.mean():.4f}, "
          f"std={J_surf_late.std():.6f}", flush=True)
    print(f"    theta_surf:   mean={theta_surf_late.mean():.4f}, "
          f"std={theta_surf_late.std():.6f}", flush=True)
    print(f"    J_core:       mean={J_core[t_late].mean():.4f}, "
          f"std={J_core[t_late].std():.6f}", flush=True)
    print(f"    theta_core:   mean={theta_core[t_late].mean():.4f}, "
          f"std={theta_core[t_late].std():.6f}", flush=True)
    is_steady = (J_surf_late.std() < 1e-3 and J_core[t_late].std() < 1e-3)
    print(f"    Steady? {is_steady}", flush=True)

    # Spatial profile at t_end
    print(f"\n  Spatial profile at t={t[-1]:.1f}:", flush=True)
    print(f"    J: min={J[:, -1].min():.4f}, "
          f"max={J[:, -1].max():.4f}, "
          f"mean={J[:, -1].mean():.4f}", flush=True)
    print(f"    theta: min={theta[:, -1].min():.4f}, "
          f"max={theta[:, -1].max():.4f}, "
          f"mean={theta[:, -1].mean():.4f}", flush=True)
    print(f"    phi: min={phi[:, -1].min():.4f}, "
          f"max={phi[:, -1].max():.4f}, "
          f"mean={phi[:, -1].mean():.4f}", flush=True)

    # Save data
    npz = OUT / "overswollen_BiT010_Da020.npz"
    np.savez(npz, t=t, x=x, J=J, theta=theta, phi=phi,
             Bi_T=Bi_T, Da=Da, N=N,
             is_steady=is_steady,
             J_surf_late_mean=float(J_surf_late.mean()),
             J_surf_late_std=float(J_surf_late.std()),
             theta_surf_late_mean=float(theta_surf_late.mean()),
             theta_surf_late_std=float(theta_surf_late.std()))
    print(f"\n  saved {npz}", flush=True)

    # Figure: 4 panels
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(PRE_DOUBLE, 6.0))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # (a) J(xi, t) heatmap
    im = ax_a.imshow(J, origin="lower", aspect="auto",
                     extent=[t[0], t[-1], x[0], x[-1]],
                     cmap="viridis")
    cbar = fig.colorbar(im, ax=ax_a)
    cbar.set_label("J", fontsize=8)
    ax_a.set_xlabel("t")
    ax_a.set_ylabel(r"$\xi$")
    ax_a.set_title("(a) J(xi, t)")

    # (b) theta(xi, t) heatmap
    im2 = ax_b.imshow(theta, origin="lower", aspect="auto",
                      extent=[t[0], t[-1], x[0], x[-1]],
                      cmap="inferno")
    cbar2 = fig.colorbar(im2, ax=ax_b)
    cbar2.set_label(r"$\theta$", fontsize=8)
    ax_b.set_xlabel("t")
    ax_b.set_ylabel(r"$\xi$")
    ax_b.set_title(r"(b) $\theta(\xi, t)$")

    # (c) Surface time series (J, theta)
    ax_c2 = ax_c.twinx()
    l1, = ax_c.plot(t, J_surf, color="C0", lw=0.8, label=r"$J_\mathrm{surf}$")
    l2, = ax_c.plot(t, J_core, color="C0", lw=0.8, ls="--",
                     label=r"$J_\mathrm{core}$")
    l3, = ax_c2.plot(t, theta_surf, color="C3", lw=0.8,
                     label=r"$\theta_\mathrm{surf}$")
    ax_c.set_xlabel("t")
    ax_c.set_ylabel("J", color="C0")
    ax_c2.set_ylabel(r"$\theta$", color="C3")
    ax_c.legend(handles=[l1, l2, l3], loc="lower right", fontsize=7)
    ax_c.set_title("(c) Surface & core time series")

    # (d) Spatial profile at t = t_end
    ax_d.plot(x, J[:, -1], "o-", color="C0",
              label=fr"J@$t$={t[-1]:.0f}", ms=2, lw=0.8)
    ax_d.plot(x, phi[:, -1] * 5, "s-", color="C2",
              label=r"$5\,\varphi$", ms=2, lw=0.8)
    ax_d_2 = ax_d.twinx()
    ax_d_2.plot(x, theta[:, -1], "^-", color="C3",
                label=r"$\theta$", ms=2, lw=0.8)
    ax_d.axhline(1.0, color="grey", ls=":", lw=0.4)
    ax_d.set_xlabel(r"$\xi$")
    ax_d.set_ylabel(r"J  /  $5\,\varphi$")
    ax_d_2.set_ylabel(r"$\theta$", color="C3")
    h1, l1_ = ax_d.get_legend_handles_labels()
    h2, l2_ = ax_d_2.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1_ + l2_, loc="best", fontsize=7)
    ax_d.set_title(f"(d) Spatial profile at t={t[-1]:.0f}")

    fig.suptitle(
        f"Over-swollen SS:  Bi_T={Bi_T}, Da={Da}  "
        f"(steady: {is_steady}, J_surf={J_surf_late.mean():.3f})",
        fontsize=10)
    fig.tight_layout()
    pdf = FIG_DIR / "iv_c_overswollen_diag.pdf"
    png = FIG_DIR / "iv_c_overswollen_diag.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf}", flush=True)
    print(f"  saved {png}", flush=True)


if __name__ == "__main__":
    main()
