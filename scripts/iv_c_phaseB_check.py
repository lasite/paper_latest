#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseB_check.py — Phase B verification + period-collapse figure.

Reads
  data/iv_c/phaseB/period_scan.npz       (this phase)
  data/iv_c/folds/S_chi_sweep.npz        (Phase P analytic limit)

Verifies
  T_PDE * Bi_T  ->  ln(theta_up / theta_lo)
                    where theta_up, theta_lo come from the F-H-R fold
                    solver evaluated at each S_chi.

Output
  data/iv_c/phaseB/phaseB_check.npz       (rel errors, PASS/FAIL flag)
  Figure/pub/iv_c_period_collapse.{pdf,png}   (collapse plot)

PASS criterion
  max relative error at smallest Bi_T < 30%   (per plan §5)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE   # type: ignore
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = _HERE.parent / "data" / "iv_c"
FIG_DIR = _HERE.parent / "Figure" / "pub"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PASS_THRESHOLD = 0.30  # 30% rel err at smallest Bi_T


def main():
    d = np.load(DATA / "phaseB" / "period_scan.npz")
    folds = np.load(DATA / "folds" / "S_chi_sweep.npz")

    Bi_T   = d["Bi_T"]
    S_chi  = d["S_chi"]
    T_grid = d["T_grid"]            # (n_BiT, n_Schi)
    is_osc = d["is_oscillating"]
    nx, ny = T_grid.shape

    # Analytic asymptote ln(theta_up/theta_lo) interpolated at our S_chi
    f_lim = interp1d(folds["S_chi"], folds["log_theta_ratio"],
                     fill_value="extrapolate", bounds_error=False)
    asym = f_lim(S_chi)             # (n_Schi,)

    # T * Bi_T  per (i, j)
    T_BiT = T_grid * Bi_T[:, None]   # broadcast Bi_T over j

    # Per S_chi, take the smallest oscillating Bi_T as the asymptote test
    print("=" * 72)
    print(" Phase B - period scaling check")
    print("=" * 72)
    print(f"  Grid: Bi_T={list(Bi_T)}   S_chi={list(S_chi)}")
    print(f"  Asymptotic prediction T*Bi_T -> ln(theta_up/theta_lo)")
    print(f"\n  S_chi  ln(t_up/t_lo)   T*Bi_T(min Bi_T)   rel err   Bi_T_used")
    print("  " + "-" * 60)

    rel_errs = []
    used_BiT = []
    used_TBiT = []
    for j, sc in enumerate(S_chi):
        col = T_BiT[:, j]
        osc_col = is_osc[:, j]
        finite = np.isfinite(col) & osc_col
        if not finite.any():
            print(f"  {sc:6.2f}  {asym[j]:10.4f}        --                  --     no oscillation")
            rel_errs.append(np.nan)
            used_BiT.append(np.nan)
            used_TBiT.append(np.nan)
            continue
        i_min = int(np.where(finite)[0][0])    # smallest Bi_T that oscillates
        meas = col[i_min]
        rel = abs(meas - asym[j]) / abs(asym[j])
        rel_errs.append(float(rel))
        used_BiT.append(float(Bi_T[i_min]))
        used_TBiT.append(float(meas))
        print(f"  {sc:6.2f}  {asym[j]:10.4f}        {meas:10.4f}        "
              f"{rel:7.2%}    {Bi_T[i_min]:.3f}")

    rel_errs = np.array(rel_errs)
    finite_errs = rel_errs[np.isfinite(rel_errs)]
    if len(finite_errs) == 0:
        print("\n  >>> Phase B FAIL - no oscillating points to test. <<<")
        sys.exit(1)
    max_err = float(np.max(finite_errs))
    mean_err = float(np.mean(finite_errs))

    PASS = max_err < PASS_THRESHOLD
    print(f"\n  max rel err: {max_err:.2%}  (threshold {PASS_THRESHOLD:.0%})")
    print(f"  mean rel err: {mean_err:.2%}")
    print(f"  >>> Phase B {'PASS' if PASS else 'FAIL'} <<<")

    np.savez(DATA / "phaseB" / "phaseB_check.npz",
             Bi_T=Bi_T, S_chi=S_chi,
             T_BiT=T_BiT, asym=asym,
             rel_errs=rel_errs, used_BiT=np.array(used_BiT),
             used_TBiT=np.array(used_TBiT),
             max_err=max_err, mean_err=mean_err,
             pass_threshold=PASS_THRESHOLD)

    # ─────────────── Figure ───────────────
    set_style()
    fig, ax = plt.subplots(figsize=(PRE_DOUBLE / 2, 3.0))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(S_chi.min(), S_chi.max())

    for j, sc in enumerate(S_chi):
        col = T_BiT[:, j]
        ok = is_osc[:, j] & np.isfinite(col)
        if ok.any():
            ax.plot(Bi_T[ok], col[ok], "o-", color=cmap(norm(sc)),
                    label=fr"$S_\chi$ = {sc:.1f}", lw=1.0, ms=4)
            # Analytic limit dashed line
            ax.axhline(asym[j], color=cmap(norm(sc)), ls=":", lw=0.7)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{Bi}_T$")
    ax.set_ylabel(r"$T_{\mathrm{PDE}}\,\mathrm{Bi}_T$")
    ax.set_title(r"Period collapse: $T\,\mathrm{Bi}_T \to \ln(\theta_{up}/\theta_{lo})$")
    ax.legend(fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    pdf = FIG_DIR / "iv_c_period_collapse.pdf"
    png = FIG_DIR / "iv_c_period_collapse.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {pdf}")
    print(f"  saved {png}")

    if not PASS:
        sys.exit(1)


if __name__ == "__main__":
    main()
