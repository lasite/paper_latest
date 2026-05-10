#!/usr/bin/env python3
"""
thermo_axis_scan.py — Stage-2 supplementary scan along the thermodynamic
axes (χ_∞, S_χ).

The hypothesis under test is that the LCST front position ξ_LCST is
*thermodynamic-limited*: it sits at the deepest ξ where θ(ξ) is high
enough to push χ(θ) past the LCST fold of the (J, θ) bistable curve.
If true, ξ_LCST should move strongly when we change parameters that
shift the fold (χ_∞ shifts the whole curve; S_χ rescales the χ–θ
slope) but barely move when we change reactant-side parameters
(D₀, Da, Bi_c) — exactly the pattern observed in fig 3f.

Outputs:
  data/fig3/thermo_axis_chi_inf.npz
  data/fig3/thermo_axis_S_chi.npz
  Figure/pub/fig3_thermo_axis.png
"""
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT, T_START, T_END
from fig3_data import (cold_J_eq, derived_from_arrays, thiele_modulus)

set_style()

DATA_DIR = (_HERE.parent / "data" / "fig3").resolve()
OUT_DIR  = (_HERE.parent / "Figure" / "pub").resolve()


SCANS = [
    dict(param="chi_inf", vals=np.array([0.45, 0.50, 0.55, 0.60, 0.65,
                                          0.70, 0.75, 0.80])),
    dict(param="S_chi",   vals=np.array([0.50, 0.65, 0.80, 0.95, 1.10,
                                          1.25, 1.40, 1.55])),
]
AMP_THRESH = 0.20


def _worker(task):
    param, val = task
    from scan_optimized import Params, simulate

    base = dict(WORKING_POINT)
    base[param] = float(val)
    try:
        p = Params(**base)
        result = simulate(p)
    except Exception as e:
        return param, val, None, f"sim_failed: {e}"

    t = result["t"]; J = result["J"]
    u = np.maximum(result["u"], 1e-12); theta = result["theta"]; x = result["x"]
    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        return param, val, None, "short_window"

    surf_amp = float(J[-1, idx].max() - J[-1, idx].min())
    if surf_amp < AMP_THRESH:
        return param, val, None, f"not_oscillating (surf={surf_amp:.3f})"

    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], base)
    theta_w = theta[:, idx]
    th_mean = float(theta_w.mean()); th_max = float(theta_w.max())
    if np.isfinite(d["xi_LCST"]):
        sw = x < d["xi_LCST"]
        th_eff = float(theta_w[sw].mean()) if sw.any() else th_mean
    else:
        th_eff = th_mean
    J_eq = float(d["J_eq"])
    zeta = thiele_modulus(base, theta_eff=th_eff, J_eq=J_eq, with_Bi_c=True)
    return param, val, dict(
        xi_peak=float(d["xi_peak"]), xi_LCST=float(d["xi_LCST"]),
        theta_mean=th_mean, theta_max=th_max, theta_eff=th_eff,
        J_eq=J_eq, surf_amp=surf_amp, zeta=float(zeta),
    ), "ok"


def run_scan(param, vals, force=False, n_workers=12):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"thermo_axis_{param}.npz"
    if path.exists() and not force:
        print(f"  Using cache: {path}")
        return dict(np.load(path))

    n = len(vals)
    keys = ["xi_peak", "xi_LCST", "theta_mean", "theta_max", "theta_eff",
            "J_eq", "surf_amp", "zeta"]
    out = {k: np.full(n, np.nan) for k in keys}
    out["vals"] = vals
    out["ok"]   = np.zeros(n, dtype=bool)
    val_to_idx = {float(v): k for k, v in enumerate(vals)}

    print(f"  Scanning {param} ∈ {vals.tolist()} (workers={n_workers})")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, (param, float(v))): float(v) for v in vals}
        for f in as_completed(futs):
            par, val, r, status = f.result()
            i = val_to_idx[float(val)]
            print(f"    {par}={val:.4f}: {status}", end="")
            if r is None:
                print()
                continue
            out["ok"][i] = True
            for k in keys:
                out[k][i] = r[k]
            print(f"  xi_LCST={r['xi_LCST']:.3f}, zeta={r['zeta']:.2f}, "
                  f"theta_eff={r['theta_eff']:.2f}")

    np.savez_compressed(path, **out)
    print(f"  Saved: {path}")
    return out


PARAM_LABEL = {"chi_inf": r"$\chi_\infty$", "S_chi": r"$S_\chi$"}
PARAM_COLOR = {"chi_inf": "#1f5fa3", "S_chi": "#a23e1c"}
PARAM_MARKER = {"chi_inf": "o", "S_chi": "s"}


def main():
    scans = {}
    for s in SCANS:
        scans[s["param"]] = run_scan(s["param"], s["vals"])

    fig, axes = plt.subplots(1, 3, figsize=(PRE_DOUBLE, 2.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.20,
                        wspace=0.42)

    # Panel (a): xi_LCST vs param/WP — show fold-position dependence
    ax = axes[0]
    for s in SCANS:
        param = s["param"]
        d = scans[param]
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        ax.plot(d["vals"][m] / WORKING_POINT[param], d["xi_LCST"][m],
                marker=PARAM_MARKER[param],
                color=PARAM_COLOR[param],
                lw=1.0, ms=5, ls="-",
                label=f"vary {PARAM_LABEL[param]}")
    ax.axvline(1.0, color="0.4", lw=0.6, ls=":")
    ax.text(1.0, 0.02, " WP", fontsize=6, color="0.3",
            ha="left", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$\beta/\beta_\mathrm{WP}$", fontsize=8)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6, loc="best", framealpha=0.9,
              handlelength=1.4, borderpad=0.3)
    add_panel_label(ax, "a")

    # Panel (b): xi_LCST vs theta_eff (the actual control variable)
    ax = axes[1]
    for s in SCANS:
        param = s["param"]
        d = scans[param]
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        ax.plot(d["theta_eff"][m], d["xi_LCST"][m],
                marker=PARAM_MARKER[param],
                color=PARAM_COLOR[param],
                lw=1.0, ms=5, ls="-",
                label=f"vary {PARAM_LABEL[param]}")
    ax.set_xlabel(r"$\langle\theta\rangle$ (swollen seg.)", fontsize=8)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    add_panel_label(ax, "b")

    # Panel (c): xi_LCST vs theta_max — peak θ that actually triggers LCST
    ax = axes[2]
    for s in SCANS:
        param = s["param"]
        d = scans[param]
        m = d["ok"].astype(bool)
        if not m.any():
            continue
        ax.plot(d["theta_max"][m], d["xi_LCST"][m],
                marker=PARAM_MARKER[param],
                color=PARAM_COLOR[param],
                lw=1.0, ms=5, ls="-",
                label=f"vary {PARAM_LABEL[param]}")
    ax.set_xlabel(r"$\theta_\mathrm{max}$", fontsize=8)
    ax.set_ylabel(r"$\xi_\mathrm{LCST}$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    add_panel_label(ax, "c")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig3_thermo_axis.pdf"
    png = OUT_DIR / "fig3_thermo_axis.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")


if __name__ == "__main__":
    main()
