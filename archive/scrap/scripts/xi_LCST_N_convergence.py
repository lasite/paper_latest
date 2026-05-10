#!/usr/bin/env python3
"""
xi_LCST_N_convergence.py — Stage-2 supplement: convergence of ξ_LCST
with grid resolution N.

The fig3/fig4 working point uses N=41 cells; that gives dx ≈ 0.024 and
the last few cell centers at ξ ∈ {0.927, 0.951, 0.976}. The observed
ξ_LCST ≈ 0.922 is suspiciously close to the second-to-last cell, which
could mean either (a) the LCST front genuinely sits at ξ ≈ 0.92
(geometry-pinned) and N=41 resolves it adequately, or (b) we are
seeing the N=41 grid quantization and ξ_LCST → 1 as N → ∞.

This script reruns the working-point simulation at N ∈ {21, 41, 81,
161} and reports ξ_LCST and ξ_peak for each, in parallel.
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
from fig3_data import derived_from_arrays

set_style()

DATA_DIR = (_HERE.parent / "data" / "fig3").resolve()
OUT_DIR  = (_HERE.parent / "Figure" / "pub").resolve()


N_VALS = [21, 41, 61, 81, 121, 161]


def _worker(N):
    from scan_optimized import Params, simulate
    p_dict = dict(WORKING_POINT)
    p_dict["N"] = int(N)
    p_dict["t_end"] = 250.0
    p_dict["n_save"] = 4000
    try:
        p = Params(**p_dict)
        result = simulate(p)
    except Exception as e:
        return N, None, f"failed: {e}"
    t = result["t"]; J = result["J"]
    u = np.maximum(result["u"], 1e-12)
    theta = result["theta"]; x = result["x"]
    idx = (t >= T_START) & (t <= T_END)
    if idx.sum() < 50:
        return N, None, "short_window"
    surf_amp = float(J[-1, idx].max() - J[-1, idx].min())
    if surf_amp < 0.20:
        return N, None, f"not_oscillating (surf={surf_amp:.3f})"
    d = derived_from_arrays(x, J[:, idx], u[:, idx], theta[:, idx], p_dict)
    return N, dict(
        N=int(N), dx=1.0/int(N),
        xi_peak=float(d["xi_peak"]),
        xi_LCST=float(d["xi_LCST"]),
        surf_amp=surf_amp,
    ), "ok"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "xi_LCST_N_convergence.npz"

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        Ns = z["N"]; xi_p = z["xi_peak"]; xi_L = z["xi_LCST"]; dxs = z["dx"]
        print(f"  Using cache: {cache}")
    else:
        print(f"  Running N ∈ {N_VALS} (parallel)")
        results = {}
        with ProcessPoolExecutor(max_workers=min(len(N_VALS), 6)) as ex:
            futs = {ex.submit(_worker, N): N for N in N_VALS}
            for f in as_completed(futs):
                N, r, status = f.result()
                print(f"    N={N}: {status}", end="")
                if r is not None:
                    print(f"  xi_LCST={r['xi_LCST']:.5f}  "
                          f"xi_peak={r['xi_peak']:.5f}  surf={r['surf_amp']:.2f}")
                    results[N] = r
                else:
                    print()
        Ns_ok = sorted(results.keys())
        Ns = np.array(Ns_ok)
        xi_p = np.array([results[N]["xi_peak"] for N in Ns_ok])
        xi_L = np.array([results[N]["xi_LCST"] for N in Ns_ok])
        dxs  = np.array([results[N]["dx"]      for N in Ns_ok])
        np.savez_compressed(cache, N=Ns, xi_peak=xi_p, xi_LCST=xi_L, dx=dxs)
        print(f"  Saved: {cache}")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.92, bottom=0.20)
    ax.plot(dxs, xi_L, "o-", color="#a23e1c", lw=1.0, ms=5,
            label=r"$\xi_\mathrm{LCST}$")
    ax.plot(dxs, xi_p, "s--", color="#1f5fa3", lw=1.0, ms=4,
            label=r"$\xi_\mathrm{peak}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$dx = 1/N$", fontsize=8)
    ax.set_ylabel(r"$\xi$", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.9,
              handlelength=1.4, borderpad=0.3)
    # Annotate N values
    for d, x_l, N in zip(dxs, xi_L, Ns):
        ax.annotate(f"N={N}", xy=(d, x_l), xytext=(2, 4),
                    textcoords="offset points", fontsize=5.5,
                    color="0.25")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig3_aux_N_convergence.pdf"
    png = OUT_DIR / "fig3_aux_N_convergence.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {pdf}")

    print("\n  Summary:")
    print(f"    {'N':>4} {'dx':>7} {'xi_peak':>9} {'xi_LCST':>9}")
    for N, dx, xp, xl in zip(Ns, dxs, xi_p, xi_L):
        print(f"    {N:>4} {dx:>7.4f} {xp:>9.5f} {xl:>9.5f}")


if __name__ == "__main__":
    main()
