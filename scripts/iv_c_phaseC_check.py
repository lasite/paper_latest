#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iv_c_phaseC_check.py — Phase C verification.

Reads
  data/iv_c/phaseC/Da_c_0D.npz       (analytic 0D Da_c)
  data/iv_c/phaseC/Da_c_pde.npz      (PDE bisection)

Verifies scaling law (iv):
    (Da_c^PDE - Da_c^0D) / Da_c^0D  =  c_1 * sqrt(Bi_T / alpha)
                                        + c_2 / Bi_c + O(ell^2)
with c_1, c_2 > 0.  We perform a linear fit of the LHS vs
sqrt(Bi_T / alpha) at fixed Bi_c.

PASS criteria
  - linear fit has positive slope
  - R^2 > 0.85
  - LHS is positive at every Bi_T  (PDE always more stable than 0D)

Outputs
  data/iv_c/phaseC/phaseC_check.npz
  Figure/pub/iv_c_onset_shift.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from style_pub import set_style, PRE_DOUBLE  # type: ignore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = _HERE.parent / "data" / "iv_c" / "phaseC"
FIG_DIR = _HERE.parent / "Figure" / "pub"
FIG_DIR.mkdir(parents=True, exist_ok=True)

R2_THRESHOLD = 0.85


def main():
    d0D = np.load(DATA / "Da_c_0D.npz")
    dPDE = np.load(DATA / "Da_c_pde.npz")

    Bi_T = dPDE["Bi_T"]
    Da_c_0D = dPDE["Da_c_0D"]      # = d0D["Da_c_0D"], confirmed at save time
    Da_c_PDE = dPDE["Da_c_pde"]
    alpha = float(d0D["alpha"])

    finite = np.isfinite(Da_c_PDE) & np.isfinite(Da_c_0D)
    if finite.sum() < 2:
        print("  >>> Phase C FAIL - fewer than 2 finite Da_c^PDE values <<<")
        sys.exit(1)

    Bi_T_f = Bi_T[finite]
    Da_c_0D_f = Da_c_0D[finite]
    Da_c_PDE_f = Da_c_PDE[finite]

    shift = (Da_c_PDE_f - Da_c_0D_f) / Da_c_0D_f         # LHS
    x_axis = np.sqrt(Bi_T_f / alpha)                      # predictor

    # Linear fit shift = m * x + b
    m, b = np.polyfit(x_axis, shift, 1)
    yhat = m * x_axis + b
    ss_res = float(np.sum((shift - yhat) ** 2))
    ss_tot = float(np.sum((shift - shift.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    all_positive = bool(np.all(shift > 0))
    PASS = (m > 0) and (R2 > R2_THRESHOLD) and all_positive

    print("=" * 70)
    print(" Phase C - onset-shift scaling check")
    print("=" * 70)
    print(f"  alpha = {alpha:.3f}")
    print(f"  {'Bi_T':>7s}  {'Da_c^0D':>9s}  {'Da_c^PDE':>9s}  "
          f"{'shift':>9s}  {'sqrt(Bi_T/alpha)':>16s}")
    for i in range(len(Bi_T_f)):
        print(f"  {Bi_T_f[i]:7.3f}  {Da_c_0D_f[i]:9.4f}  "
              f"{Da_c_PDE_f[i]:9.4f}  {shift[i]:+8.3f}  "
              f"{x_axis[i]:16.3f}")
    print()
    print(f"  Linear fit:   shift = {m:+.4f} * sqrt(Bi_T/alpha) + {b:+.4f}")
    print(f"  R^2 = {R2:.4f}")
    print(f"  All shifts positive: {all_positive}")
    print(f"  >>> Phase C {'PASS' if PASS else 'FAIL'} <<<")

    np.savez(DATA / "phaseC_check.npz",
             Bi_T=Bi_T_f, Da_c_0D=Da_c_0D_f, Da_c_PDE=Da_c_PDE_f,
             shift=shift, x_axis=x_axis,
             slope=m, intercept=b, R2=R2,
             all_positive=all_positive,
             pass_threshold=R2_THRESHOLD)

    # ── Figure ───────────────────────────────────────────
    set_style()
    fig, ax = plt.subplots(figsize=(PRE_DOUBLE / 2, 3.0))
    ax.plot(x_axis, shift, "o", ms=6, label="PDE bisection")
    xs = np.linspace(0, x_axis.max() * 1.1, 50)
    ax.plot(xs, m * xs + b, "-", lw=1.0, color="C1",
            label=fr"fit: $c_1$={m:.2f}, $R^2$={R2:.2f}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(r"$\sqrt{\mathrm{Bi}_T / \alpha}$")
    ax.set_ylabel(r"$(\mathrm{Da}_c^\mathrm{PDE} - \mathrm{Da}_c^{0D})"
                  r"/\mathrm{Da}_c^{0D}$")
    ax.set_title("Phase C: onset-shift scaling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pdf = FIG_DIR / "iv_c_onset_shift.pdf"
    png = FIG_DIR / "iv_c_onset_shift.png"
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {pdf}")
    print(f"  saved {png}")

    if not PASS:
        sys.exit(1)


if __name__ == "__main__":
    main()
