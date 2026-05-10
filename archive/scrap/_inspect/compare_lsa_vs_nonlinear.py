#!/usr/bin/env python3
"""
compare_lsa_vs_nonlinear.py — Agreement test between linear stability
(0D Hopf-unstable indicator from hopf_boundary.py) and the nonlinear
regime classification (fig4_data.build_grid). For each (Bi_T, S_chi) and
(Bi_T, Da) cell, check whether:

  LSA-Hopf-unstable  ⇔  nonlinear-oscillatory

Cross-tab:                       LSA stable    LSA Hopf-unstable
       nonlinear steady             A                  B  (subcritical / unresolved)
       nonlinear oscillating        C  (subcritical?)  D

Reports counts and the worst (parameter-set, regime) outliers.
"""
import os, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from fig4_data import (
    BI_T_VALS, S_CHI_VALS, DA_VALS, _grid_path,
    REG_FAILED, REG_STEADY_COLD, REG_BULK_HOPF, REG_LCST_FRONT,
    REG_GLOBAL_COLLAPSE, REG_STEADY_COLLAPSED, REG_STEADY_FRONT,
    REGIME_NAMES,
)
from hopf_boundary import hopf_grid


OSC_REGIMES   = {REG_BULK_HOPF, REG_LCST_FRONT, REG_GLOBAL_COLLAPSE}
STEADY_REGIMES = {REG_STEADY_COLD, REG_STEADY_COLLAPSED, REG_STEADY_FRONT}


def compare_one(name, px, xv, py, yv):
    print(f"\n=== {name} ({px} × {py}) ===")
    g = np.load(_grid_path(px, yv_param := py))
    h = hopf_grid(px, xv, py, yv)
    reg  = g["regime"]
    re_c = h["re_max_complex"]
    nx, ny = len(xv), len(yv)
    assert reg.shape == (ny, nx), f"shape mismatch {reg.shape} vs {(ny,nx)}"

    # LSA Hopf-unstable: re_max_complex > 0 (real part of complex eigenvalue pair > 0)
    lsa_hopf = np.isfinite(re_c) & (re_c > 0.0)
    nl_osc = np.isin(reg, list(OSC_REGIMES))
    nl_steady = np.isin(reg, list(STEADY_REGIMES))
    nl_failed = (reg == REG_FAILED)

    # Two-by-two table
    A = int(((~lsa_hopf) & nl_steady).sum())
    B = int((lsa_hopf & nl_steady).sum())
    C = int(((~lsa_hopf) & nl_osc).sum())
    D = int((lsa_hopf & nl_osc).sum())
    F = int(nl_failed.sum())
    n = nx * ny

    print(f"  total cells: {n}")
    print(f"  ┌──────────────────────┬───────────────┬───────────────────┐")
    print(f"  │                      │   LSA stable  │  LSA Hopf-unstable │")
    print(f"  ├──────────────────────┼───────────────┼───────────────────┤")
    print(f"  │ nonlinear steady     │ {A:>13d} │ {B:>17d} │")
    print(f"  │ nonlinear oscillating│ {C:>13d} │ {D:>17d} │")
    print(f"  └──────────────────────┴───────────────┴───────────────────┘")
    print(f"  failed: {F}")
    agreement = (A + D) / max(n - F, 1) * 100
    print(f"  Agreement (excl. failed): {agreement:.1f}%")
    print(f"  Disagreement:")
    print(f"    cells where LSA says unstable but nonlinear stays steady : B={B}  "
          f"({B/max(n-F,1)*100:.1f}%)")
    print(f"    cells where LSA says stable but nonlinear oscillates     : C={C}  "
          f"({C/max(n-F,1)*100:.1f}%)")

    # Spotlight: show the worst outliers
    j_amp = g["J_amp_max"]
    if B > 0:
        print(f"\n  B-cells (LSA unstable, nonlinear steady):")
        ys, xs = np.where(lsa_hopf & nl_steady)
        amps = j_amp[ys, xs]
        regs = reg[ys, xs]
        print(f"    nonlinear J_amp distribution:  "
              f"min={np.nanmin(amps):.4f}  median={np.nanmedian(amps):.4f}  "
              f"max={np.nanmax(amps):.4f}")
        # By regime
        from collections import Counter
        by_reg = Counter(int(r) for r in regs)
        for code, n in by_reg.most_common():
            print(f"    regime={REGIME_NAMES.get(code,'?'):>20s}: {n}")
    if C > 0:
        print(f"\n  C-cells (LSA stable, nonlinear oscillates):")
        ys, xs = np.where((~lsa_hopf) & nl_osc)
        amps = j_amp[ys, xs]
        re_vals = re_c[ys, xs]
        regs = reg[ys, xs]
        print(f"    J_amp distribution:    "
              f"min={np.nanmin(amps):.4f}  median={np.nanmedian(amps):.4f}  "
              f"max={np.nanmax(amps):.4f}")
        # How far below Hopf onset are these cells?
        finite = np.isfinite(re_vals)
        if finite.any():
            print(f"    re_complex distribution: "
                  f"min={np.nanmin(re_vals[finite]):+.3e}  "
                  f"median={np.nanmedian(re_vals[finite]):+.3e}  "
                  f"max={np.nanmax(re_vals[finite]):+.3e}  "
                  f"({(~finite).sum()} -inf cells)")
        from collections import Counter
        by_reg = Counter(int(r) for r in regs)
        for code, n in by_reg.most_common():
            print(f"    regime={REGIME_NAMES.get(code,'?'):>20s}: {n}")


def main():
    compare_one("Bi_T × S_chi", "Bi_T", BI_T_VALS, "S_chi", S_CHI_VALS)
    compare_one("Bi_T × Da",    "Bi_T", BI_T_VALS, "Da",    DA_VALS)


if __name__ == "__main__":
    main()
