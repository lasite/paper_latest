#!/usr/bin/env python3
"""
thiele_collapse.py — Stage-2 universal scaling for the LCST front.

We want to test whether ξ_LCST (and ξ_peak) collapse to one universal
curve as a function of the swollen-branch Thiele modulus ζ. To do so
we run *several* 1D scans across parameters that all enter ζ
multiplicatively but differ in physical role (D₀, Da, Bi_c, plus Bi_T
and α for the orthogonal axes that should NOT appear in ζ if the
prediction is correct).

For each scan point we record:
  ξ_peak, ξ_LCST   — front diagnostics from derived_from_arrays
  ⟨θ⟩, θ_max       — for the Thiele θ_eff
  J_eq             — cold-bath equilibrium (needed for φ_eq)
  ζ (with Bi_c correction) — Thiele modulus computed from ⟨θ⟩

Outputs:
  data/fig3/thiele_collapse_<param>.npz
  Figure/pub/fig3_thiele_collapse.png  — the collapse plot
"""
import os
import sys
from pathlib import Path

# Pin BLAS *before* numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT, T_START, T_END
from fig3_data import (cold_J_eq, derived_from_arrays, thiele_modulus,
                       FIG_DIR as FIG3_DIR)

set_style()

DATA_DIR = (_HERE.parent / "data" / "fig3").resolve()
OUT_DIR  = (_HERE.parent / "Figure" / "pub").resolve()


# Each scan is meant to STAY in the LCST-front oscillation regime, so
# we choose the range conservatively around the working point. Large
# steps would tip the system into a frozen-front or steady-cold cell
# and ξ_LCST becomes meaningless.
SCANS = [
    # Primary: parameters that ENTER ζ
    dict(param="D0",   vals=np.array([0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5,
                                      3.0, 3.6])),
    dict(param="Da",   vals=np.array([2.0, 2.6, 3.2, 4.0, 5.0, 6.5, 8.5,
                                      11.0])),
    dict(param="Bi_c", vals=np.array([0.30, 0.45, 0.60, 0.75, 0.90, 1.10,
                                      1.35, 1.65, 2.00])),
    # Orthogonal: parameters that should NOT appear in ζ  if the
    # prediction is right (used as collapse-quality check).
    dict(param="Bi_T", vals=np.array([0.06, 0.07, 0.08, 0.09, 0.10,
                                      0.11, 0.12, 0.13])),
    dict(param="alpha",vals=np.array([0.015, 0.02, 0.025, 0.03, 0.04,
                                      0.05, 0.06])),
]

AMP_THRESH = 0.20


def _run_one(param, val):
    """Simulate one parameter point and return diagnostics + Thiele ζ."""
    from scan_optimized import Params, simulate

    base = dict(WORKING_POINT)
    base[param] = float(val)
    p = Params(**base)
    try:
        result = simulate(p)
    except Exception as e:
        print(f"    {param}={val}: FAILED — {e}")
        return None

    t = result["t"]; J = result["J"]; u = np.maximum(result["u"], 1e-12)
    theta = result["theta"]; x = result["x"]
    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        return None

    surf_amp = float(J[-1, idx].max() - J[-1, idx].min())
    if surf_amp < AMP_THRESH:
        return None

    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], base)
    theta_w = theta[:, idx]
    th_mean = float(theta_w.mean())
    th_max  = float(theta_w.max())
    # Use mean(θ) on the swollen segment (ξ < ξ_LCST) for a more honest
    # θ_eff than 0D SS — the 0D SS overestimates how cool the swollen
    # interior actually is.
    if np.isfinite(d["xi_LCST"]):
        sw_mask = x < d["xi_LCST"]
        if sw_mask.any():
            th_eff = float(theta_w[sw_mask].mean())
        else:
            th_eff = th_mean
    else:
        th_eff = th_mean

    J_eq = float(d["J_eq"])
    zeta = thiele_modulus(base, theta_eff=th_eff, J_eq=J_eq,
                          with_Bi_c=True)
    zeta_no_Bic = thiele_modulus(base, theta_eff=th_eff, J_eq=J_eq,
                                 with_Bi_c=False)

    return dict(
        xi_peak=float(d["xi_peak"]),
        xi_LCST=float(d["xi_LCST"]),
        theta_mean=th_mean,
        theta_max=th_max,
        theta_eff=th_eff,
        J_eq=J_eq,
        surf_amp=surf_amp,
        zeta=float(zeta),
        zeta_no_Bic=float(zeta_no_Bic),
    )


def _cache_path(param):
    return DATA_DIR / f"thiele_collapse_{param}.npz"


def load_or_run(param, vals, force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(param)
    if path.exists() and not force:
        print(f"  Using cache: {path}")
        return dict(np.load(path))

    print(f"  Scanning {param} ∈ {vals.tolist()} ...")
    n = len(vals)
    keys = ["xi_peak", "xi_LCST", "theta_mean", "theta_max", "theta_eff",
            "J_eq", "surf_amp", "zeta", "zeta_no_Bic"]
    out = {k: np.full(n, np.nan) for k in keys}
    out["vals"] = vals
    out["ok"]   = np.zeros(n, dtype=bool)

    for k, v in enumerate(vals):
        print(f"    [{k+1}/{n}] {param}={v:.4f}", end=" ... ")
        r = _run_one(param, v)
        if r is None:
            print("skipped")
            continue
        out["ok"][k] = True
        for kk in keys:
            out[kk][k] = r[kk]
        print(f"xi_LCST={r['xi_LCST']:.3f}, zeta={r['zeta']:.3f}, "
              f"theta_eff={r['theta_eff']:.2f}")

    np.savez_compressed(path, **out)
    print(f"  Saved: {path}")
    return out


PARAM_LABEL = {
    "Bi_T":  r"$\mathrm{Bi}_T$",
    "Bi_c":  r"$\mathrm{Bi}_c$",
    "Da":    r"$\mathrm{Da}$",
    "alpha": r"$\alpha$",
    "D0":    r"$D_0$",
}
PARAM_MARKER = {"Bi_T": "o", "D0": "s", "alpha": "^",
                "Da": "v", "Bi_c": "D"}
PARAM_COLOR  = {"Bi_T": "#1f5fa3", "D0": "#a23e1c", "alpha": "#137a73",
                "Da":   "#7f3f98", "Bi_c": "#c89a2c"}


def main():
    scans = {}
    for s in SCANS:
        scans[s["param"]] = load_or_run(s["param"], s["vals"])

    fig, axes = plt.subplots(1, 3, figsize=(PRE_DOUBLE, 2.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.20,
                        wspace=0.38)

    # Panel (a): xi_LCST vs zeta (with Bi_c correction)
    ax = axes[0]
    for s in SCANS:
        param = s["param"]
        d = scans[param]
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        ax.plot(d["zeta"][m], d["xi_LCST"][m],
                marker=PARAM_MARKER.get(param, "o"),
                color=PARAM_COLOR.get(param, "0.3"),
                lw=0.9, ms=4.5, alpha=0.9, ls="-",
                label=f"vary {PARAM_LABEL.get(param, param)}")
    ax.set_xlabel(r"$\zeta$ (Thiele, swollen branch)", fontsize=8)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.set_xscale("log")
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="best", framealpha=0.9,
              handlelength=1.4, borderpad=0.3, ncol=1, labelspacing=0.25)
    add_panel_label(ax, "a")

    # Panel (b): xi_peak vs zeta — same test for the peak boundary
    ax = axes[1]
    for s in SCANS:
        param = s["param"]
        d = scans[param]
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        ax.plot(d["zeta"][m], d["xi_peak"][m],
                marker=PARAM_MARKER.get(param, "o"),
                color=PARAM_COLOR.get(param, "0.3"),
                lw=0.9, ms=4.5, alpha=0.9, ls="-",
                label=f"vary {PARAM_LABEL.get(param, param)}")
    ax.set_xlabel(r"$\zeta$", fontsize=8)
    ax.set_ylabel(r"$\xi_\mathrm{peak}$", fontsize=8)
    ax.set_xscale("log")
    ax.tick_params(labelsize=6.5)
    add_panel_label(ax, "b")

    # Panel (c): xi_LCST vs ζ on the SAME axes; plot the fitted
    # power-law model (1 - a/ζ^β) against the data to see how well it
    # collapses. Only data from D0/Da/Bi_c are used for the fit; Bi_T/α
    # data are overlaid as a validation set.
    ax = axes[2]
    fit_x, fit_y = [], []
    for s in SCANS:
        param = s["param"]
        d = scans[param]
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        if param in ("D0", "Da", "Bi_c"):
            fit_x.extend(d["zeta"][m].tolist())
            fit_y.extend(d["xi_LCST"][m].tolist())
        ax.plot(d["zeta"][m], d["xi_LCST"][m],
                marker=PARAM_MARKER.get(param, "o"),
                color=PARAM_COLOR.get(param, "0.3"),
                lw=0.0, ms=4.5, alpha=0.9, ls="",
                label=f"vary {PARAM_LABEL.get(param, param)}")
    fit_x = np.asarray(fit_x); fit_y = np.asarray(fit_y)
    if len(fit_x) >= 5:
        # Fit ξ_LCST = 1 - a / ζ^β by least squares on log(1-ξ_LCST) vs log ζ
        ok = (fit_y < 0.999) & (fit_x > 0)
        ly = np.log(np.maximum(1.0 - fit_y[ok], 1e-6))
        lx = np.log(fit_x[ok])
        beta, c = np.polyfit(lx, ly, 1)
        a = np.exp(c)
        zp = np.geomspace(fit_x[ok].min() * 0.8, fit_x[ok].max() * 1.2, 100)
        ax.plot(zp, 1 - a * zp**beta, "k--", lw=1.0,
                label=fr"$1 - {a:.2f}\,\zeta^{{{beta:.2f}}}$")
        print(f"\n  Fit on D0/Da/Bi_c: ξ_LCST ≈ 1 - {a:.3f}·ζ^{beta:.3f}")
        # collapse quality: residual
        pred = 1 - a * fit_x[ok]**beta
        rms = float(np.sqrt(np.mean((fit_y[ok] - pred)**2)))
        print(f"  RMS residual on fit set: {rms:.4f}")
    ax.set_xlabel(r"$\zeta$", fontsize=8)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.set_xscale("log")
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9,
              handlelength=1.6, borderpad=0.3, labelspacing=0.25,
              ncol=1)
    add_panel_label(ax, "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig3_thiele_collapse.pdf"
    png = OUT_DIR / "fig3_thiele_collapse.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
