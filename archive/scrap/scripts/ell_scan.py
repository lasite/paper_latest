#!/usr/bin/env python3
"""
ell_scan.py — Stage-3 supplement: how does ℓ (Cahn length / interface
regularization) affect the dynamics?

The N convergence test showed that the apparent "halo" between ξ_peak
and ξ_LCST shrinks as N increases (dx → ℓ from above). To decide
whether ℓ is a cosmetic numerical regularization (results invariant
to ℓ as long as dx ≪ ℓ) or an essential physical parameter (dynamics
change with ℓ), this script sweeps ℓ at fixed N=41 and records:

  ξ_LCST, ξ_peak, halo width, surface period, surface amplitude,
  mean ⟨θ⟩, max θ.

If all of these are roughly invariant under ℓ ∈ [0.005, 0.04], ℓ is
cosmetic and the cleanest fix for the N convergence stiffness issue
is to RAISE ℓ to ~0.03 (so dx ≈ ℓ at N=41) — no need to switch to
Chebyshev or AMR.
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
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, add_panel_label, PRE_DOUBLE
from fig2_data import WORKING_POINT, T_START, T_END
from fig3_data import derived_from_arrays

set_style()

DATA_DIR = (_HERE.parent / "data" / "fig3").resolve()
OUT_DIR  = (_HERE.parent / "Figure" / "pub").resolve()


# Span of ℓ values
ELL_VALS = np.array([0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040])
AMP_THRESH = 0.20


def _worker(ell):
    from scan_optimized import Params, simulate
    p_dict = dict(WORKING_POINT)
    p_dict["ell"] = float(ell)
    try:
        p = Params(**p_dict)
        result = simulate(p)
    except Exception as e:
        return ell, None, f"sim_failed: {e}"
    t = result["t"]; J = result["J"]
    u = np.maximum(result["u"], 1e-12); theta = result["theta"]; x = result["x"]
    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        return ell, None, "short_window"

    surf = J[-1, idx]
    surf_amp = float(surf.max() - surf.min())
    if surf_amp < AMP_THRESH:
        return ell, None, f"not_oscillating (amp={surf_amp:.3f})"

    # Period from surface peaks
    s = surf - surf.mean()
    pk, _ = find_peaks(s, prominence=0.1 * surf_amp, distance=3)
    tt = t[idx]
    period = float(np.median(np.diff(tt[pk]))) if len(pk) >= 2 else float("nan")

    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], p_dict)
    th_mean = float(theta[:, idx].mean())
    th_max  = float(theta[:, idx].max())

    return ell, dict(
        xi_peak=float(d["xi_peak"]),
        xi_LCST=float(d["xi_LCST"]),
        halo=float(d["xi_LCST"] - d["xi_peak"]),
        surf_amp=surf_amp,
        period=period,
        theta_mean=th_mean,
        theta_max=th_max,
    ), "ok"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "ell_scan.npz"

    print(f"  Sweeping ℓ ∈ {ELL_VALS.tolist()} (parallel {len(ELL_VALS)} workers)")
    n = len(ELL_VALS)
    keys = ["xi_peak", "xi_LCST", "halo", "surf_amp", "period",
            "theta_mean", "theta_max"]
    out = {k: np.full(n, np.nan) for k in keys}
    out["ell"] = ELL_VALS
    out["ok"]  = np.zeros(n, dtype=bool)
    val_to_idx = {float(v): k for k, v in enumerate(ELL_VALS)}

    with ProcessPoolExecutor(max_workers=min(n, 8)) as ex:
        futs = {ex.submit(_worker, float(v)): float(v) for v in ELL_VALS}
        for f in as_completed(futs):
            ell, r, status = f.result()
            i = val_to_idx[float(ell)]
            print(f"    ℓ={ell:.4f}: {status}", end="")
            if r is None:
                print()
                continue
            out["ok"][i] = True
            for k in keys:
                out[k][i] = r[k]
            print(f"  xi_LCST={r['xi_LCST']:.4f}  halo={r['halo']:.4f}  "
                  f"T={r['period']:.2f}  amp={r['surf_amp']:.2f}")

    np.savez_compressed(cache, **out)
    print(f"\n  Saved: {cache}")

    # ── Figure: 4 panels ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(PRE_DOUBLE, 2.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.22,
                        wspace=0.45)

    m = out["ok"]
    e = out["ell"][m]
    dx_N41 = 1.0 / 41.0

    # (a) ξ_LCST and ξ_peak vs ℓ
    ax = axes[0]
    ax.plot(e, out["xi_LCST"][m], "o-", color="#a23e1c", lw=1.0, ms=4.5,
            label=r"$\xi_\mathrm{LCST}$")
    ax.plot(e, out["xi_peak"][m], "s-", color="#1f5fa3", lw=1.0, ms=4,
            label=r"$\xi_\mathrm{peak}$")
    ax.axvline(dx_N41, color="0.5", lw=0.8, ls=":")
    ax.text(dx_N41, 0.94, " $dx_{N=41}$", fontsize=5.5, color="0.4",
            ha="left", va="top", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=8)
    ax.set_ylabel(r"$\xi$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6, loc="best", framealpha=0.9, handlelength=1.4,
              borderpad=0.3)
    add_panel_label(ax, "a")

    # (b) halo width vs ℓ — should grow ~ ℓ if ℓ controls front
    ax = axes[1]
    ax.plot(e, out["halo"][m], "o-", color="#7f3f98", lw=1.0, ms=4.5)
    # reference line ~ ℓ
    if len(e) >= 2:
        c = float(out["halo"][m][-1] / e[-1])
        ax.plot(e, c * e, "k--", lw=0.7, alpha=0.5, label=r"$\propto\ell$")
        ax.legend(fontsize=6, loc="best", framealpha=0.9)
    ax.axhline(dx_N41, color="0.5", lw=0.6, ls=":")
    ax.text(0.02, dx_N41, "$dx$ ", fontsize=5.5, color="0.4",
            ha="left", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=8)
    ax.set_ylabel(r"halo $=\xi_\mathrm{LCST}-\xi_\mathrm{peak}$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    add_panel_label(ax, "b")

    # (c) surface period vs ℓ
    ax = axes[2]
    ax.plot(e, out["period"][m], "o-", color="#137a73", lw=1.0, ms=4.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=8)
    ax.set_ylabel(r"surface period $T$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    # Annotate range
    if m.sum() >= 2:
        rng = float(np.ptp(out["period"][m]))
        avg = out["period"][m].mean()
        ax.text(0.02, 0.98, f"range = {rng:.2f}\nmean = {avg:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=5.5,
                bbox=dict(facecolor="white", edgecolor="0.5",
                          boxstyle="round,pad=0.2", lw=0.4))
    add_panel_label(ax, "c")

    # (d) surface amplitude vs ℓ
    ax = axes[3]
    ax.plot(e, out["surf_amp"][m], "o-", color="#c89a2c", lw=1.0, ms=4.5,
            label="amp")
    ax2 = ax.twinx()
    ax2.plot(e, out["theta_max"][m], "s--", color="#a23e1c", lw=0.9, ms=3.5,
             label=r"$\theta_\mathrm{max}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=8)
    ax.set_ylabel(r"surface $\Delta J$", fontsize=8, color="#c89a2c")
    ax.tick_params(axis="y", labelcolor="#c89a2c", labelsize=6)
    ax2.set_ylabel(r"$\theta_\mathrm{max}$", fontsize=8, color="#a23e1c")
    ax2.tick_params(axis="y", labelcolor="#a23e1c", labelsize=6)
    ax.tick_params(axis="x", labelsize=6.5)
    add_panel_label(ax, "d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig3_ell_scan.pdf"
    png = OUT_DIR / "fig3_ell_scan.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")

    # Verdict
    print("\n  ── Verdict ──")
    if m.sum() < 3:
        print("    Too few successful points to decide.")
        return
    rng_xL  = float(np.ptp(out["xi_LCST"][m]))
    rng_T   = float(np.ptp(out["period"][m]))
    rng_amp = float(np.ptp(out["surf_amp"][m]))
    avg_T   = float(out["period"][m].mean())
    avg_amp = float(out["surf_amp"][m].mean())
    print(f"    ℓ ∈ [{e.min():.4f}, {e.max():.4f}]  ({len(e)} points)")
    print(f"    ξ_LCST range:    {rng_xL:.4f}  ({rng_xL/0.92*100:.1f}% of value)")
    print(f"    Period range:    {rng_T:.2f}  ({rng_T/avg_T*100:.1f}% of mean)")
    print(f"    Amplitude range: {rng_amp:.3f}  ({rng_amp/avg_amp*100:.1f}% of mean)")
    if rng_xL < 0.03 and rng_T/avg_T < 0.10 and rng_amp/avg_amp < 0.10:
        print("    → ℓ is COSMETIC: dynamics invariant under ℓ.")
        print("    Recommendation: raise ℓ to ~0.03 + keep N=41.")
    else:
        print("    → ℓ is ESSENTIAL: dynamics depend on ℓ.")
        print("    Recommendation: switch to Chebyshev collocation or AMR.")


if __name__ == "__main__":
    main()
