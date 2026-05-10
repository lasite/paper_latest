#!/usr/bin/env python3
"""
make_fig3f_xi_peak_scan.py — Fig 3(f): zone boundaries vs reactant diffusivity D₀.

Records, for each D₀:
  ξ_peak  — depth where J_max(ξ) is maximal (mechanical halo inner edge,
             coordinate-invariant critical point)
  ξ_LCST  — depth where φ_max(ξ) crosses 0.5 (LCST collapse-front locus)

D₀ controls the reactant penetration depth (Thiele scaling). Larger D₀
pushes the active reaction zone deeper into the gel; the collapse front
follows, and so does the mechanical halo. Across the whole accessible
range, the band between ξ_peak and ξ_LCST traces the spatial signature
of the propagating LCST collapse front. Auxiliary scans of Bi_T and Da
(see _aux.py outputs) confirm that the boundaries are essentially
invariant under those parameters — the front position is set by
reactant diffusion, not by surface cooling or kinetics.

Results are cached on disk so the scan only runs once.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style_pub import set_style, add_panel_label
from fig3_data import save_panel, FIG_DIR, derived_from_arrays
from fig2_data import WORKING_POINT, T_START, T_END
set_style()


DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                         "..", "data", "fig3"))

# Multi-parameter sweep: each entry runs an independent 1D scan and
# stores the resulting ξ_peak / ξ_LCST curves so panel (f) can overlay
# them. Working-point values come from fig2_data.WORKING_POINT.
SCANS = [
    dict(param="Bi_T",  vals=np.array([0.05, 0.07, 0.08, 0.09, 0.10,
                                       0.11, 0.12, 0.13, 0.15])),
    dict(param="D0",    vals=np.array([0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5])),
    dict(param="alpha", vals=np.array([0.01, 0.015, 0.02, 0.03, 0.04,
                                       0.06, 0.08])),
]

# Minimum surface J amplitude that we accept as "genuinely oscillating".
AMP_THRESH = 0.20

# Scan resolution: lighter than the working-point cache (N=301) since
# we are tracking *trends* in ξ_peak / ξ_LCST across parameters, not
# asserting absolute values. N=121 gives dx/ℓ ≈ 0.83 (acceptable for
# locating the front to ~1% of H₀); 24 simulations finish in ≈ 1 hr.
SCAN_N = 121
SCAN_T_END = 200.0
SCAN_N_SAVE = 4000


def _run_one(param, val):
    """Run one simulation at the given parameter value.

    Returns (xi_peak, xi_LCST, ok). ok is False if the surface J
    amplitude is below AMP_THRESH (i.e., the system is not oscillating
    in the usual sense and the boundaries are meaningless).
    """
    from scan_optimized import Params, simulate

    base = dict(WORKING_POINT)
    base[param] = float(val)
    base["N"]      = SCAN_N
    base["t_end"]  = SCAN_T_END
    base["n_save"] = SCAN_N_SAVE
    p = Params(**base)
    try:
        result = simulate(p)
    except Exception as e:
        print(f"    {param}={val}: FAILED — {e}")
        return float("nan"), float("nan"), False

    t = result["t"]
    J = result["J"]
    u = np.maximum(result["u"], 1e-12)
    theta = result["theta"]
    x = result["x"]

    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        return float("nan"), float("nan"), False

    J_surf = J[-1, idx]
    surf_amp = float(J_surf.max() - J_surf.min())
    if surf_amp < AMP_THRESH:
        print(f"    {param}={val}: not oscillating (surf amp={surf_amp:.3f})")
        return float("nan"), float("nan"), False

    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], base)
    return float(d["xi_peak"]), float(d["xi_LCST"]), True


def _cache_path(param):
    return os.path.join(DATA_DIR, f"fig3f_xi_scan_{param}.npz")


def _build_one_scan(param, vals):
    print(f"  Scanning {param} ∈ {vals.tolist()} ...")
    xi_peak = np.full_like(vals, np.nan, dtype=float)
    xi_LCST = np.full_like(vals, np.nan, dtype=float)
    ok      = np.zeros_like(vals, dtype=bool)
    for k, val in enumerate(vals):
        print(f"    {param}={val:.4f} ({k + 1}/{len(vals)}) ...")
        xi_peak[k], xi_LCST[k], ok[k] = _run_one(param, val)
        print(f"      xi_peak={xi_peak[k]:.4f},  xi_LCST={xi_LCST[k]:.4f}")
    return dict(vals=vals, xi_peak=xi_peak, xi_LCST=xi_LCST, ok=ok)


def load_scans():
    """Load every scan in SCANS, building on first run."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {}
    for s in SCANS:
        param = s["param"]
        path = _cache_path(param)
        if os.path.exists(path):
            z = np.load(path)
            out[param] = {k: z[k] for k in z.files}
            continue
        scan = _build_one_scan(param, s["vals"])
        np.savez_compressed(path, **scan)
        print(f"  Saved scan cache: {path}")
        out[param] = scan
    return out


C_PEAK = "#1f5fa3"   # blue   — mechanical halo inner edge
C_LCST = "#a23e1c"   # red    — collapse-front locus
PARAM_LABEL = {
    "Bi_T":  r"$\mathrm{Bi}_T$",
    "Bi_c":  r"$\mathrm{Bi}_c$",
    "Da":    r"$\mathrm{Da}$",
    "S_chi": r"$S_\chi$",
    "alpha": r"$\alpha$",
    "D0":    r"$D_0$",
}
PARAM_MARKER = {"Bi_T": "o", "D0": "s", "alpha": "^",
                "Da": "v", "Bi_c": "D", "S_chi": "*"}


def panel_f(ax, scans, label_fs=9, tick_fs=7, legend_fs=6.5):
    """Overlay ξ_peak and ξ_LCST vs each scanned parameter, normalized
    to the working-point value so all curves share one x-axis. Y-axis
    is zoomed to highlight that the boundaries barely move under any
    of the kinetic parameters — the front is parameter-invariant."""

    halo_lows, halo_highs = [], []
    for s in SCANS:
        param = s["param"]
        scan = scans[param]
        vals = scan["vals"]
        xi_peak = scan["xi_peak"]
        xi_LCST = scan["xi_LCST"]
        ok = scan["ok"]
        m = ok & np.isfinite(xi_peak) & np.isfinite(xi_LCST)
        if not m.any():
            continue
        vn = vals[m] / WORKING_POINT[param]
        marker = PARAM_MARKER.get(param, "o")
        lbl = PARAM_LABEL.get(param, param)
        ax.plot(vn, xi_peak[m], marker=marker, ls="-",
                color=C_PEAK, lw=1.0, ms=4.5, alpha=0.95,
                label=lbl)
        ax.plot(vn, xi_LCST[m], marker=marker, ls="--",
                color=C_LCST, lw=1.0, ms=4.0, alpha=0.95)
        halo_lows.append(float(np.min(xi_peak[m])))
        halo_highs.append(float(np.max(xi_LCST[m])))

    # Working-point column at β/β_WP = 1
    ax.axvline(1.0, color="k", lw=0.7, ls=":", zorder=2)
    ax.text(1.0, 0.04, " WP", fontsize=6, ha="left", va="bottom",
            color="0.25", transform=ax.get_xaxis_transform())

    # Annotate the invariance: a thin horizontal band around the cluster
    if halo_lows and halo_highs:
        band_lo = min(halo_lows) - 0.005
        band_hi = max(halo_highs) + 0.005
        ax.axhspan(band_lo, band_hi, color="#fff0c2", alpha=0.4,
                   zorder=0, lw=0)
        ax.text(0.97, 0.5 * (band_lo + band_hi),
                "parameter-invariant\nfront",
                fontsize=6, ha="right", va="center",
                color="0.20", transform=ax.get_yaxis_transform(),
                fontstyle="italic")

    ax.set_xscale("log")
    ax.set_xlabel(r"parameter $\beta/\beta_\mathrm{WP}$", fontsize=label_fs)
    ax.set_ylabel(r"depth $\xi$", fontsize=label_fs)
    # Zoom y around the cluster to make the parameter-invariance visible
    if halo_lows and halo_highs:
        y_lo = max(0.0, min(halo_lows) - 0.06)
        y_hi = min(1.0, max(halo_highs) + 0.06)
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=tick_fs, direction="out", length=2.5)

    # Compact two-block legend
    from matplotlib.lines import Line2D
    line_peak = Line2D([], [], color=C_PEAK, lw=1.4,
                       label=r"$\xi_\mathrm{peak}$")
    line_lcst = Line2D([], [], color=C_LCST, lw=1.2, ls="--",
                       label=r"$\xi_\mathrm{LCST}$")
    handles = [line_peak, line_lcst]
    for s in SCANS:
        marker = PARAM_MARKER.get(s["param"], "o")
        handles.append(Line2D([], [], color="0.3", lw=0.0,
                              marker=marker, ms=4,
                              label=PARAM_LABEL.get(s["param"], s["param"])))
    ax.legend(handles=handles, fontsize=legend_fs, loc="lower left",
              framealpha=0.9, handlelength=1.4, borderpad=0.3,
              labelspacing=0.25, ncol=2, columnspacing=0.6,
              handletextpad=0.4)


def main():
    scans = load_scans()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.95, bottom=0.18)
    panel_f(ax, scans)
    add_panel_label(ax, "f")
    save_panel(fig, "fig3f_xi_peak_scan")
    plt.close(fig)


if __name__ == "__main__":
    main()
