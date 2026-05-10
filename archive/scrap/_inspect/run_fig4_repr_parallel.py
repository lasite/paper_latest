#!/usr/bin/env python3
"""
run_fig4_repr_parallel.py — kick off the 3 representative_runs for fig4
in parallel (instead of the serial loop inside fig4_data).

Each is one N=301, t_end=200 simulation (~5-15 min). 3 in parallel
finishes in ~15 min wall.
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def _worker(args):
    lbl, p_dict = args
    from fig4_data import (
        SIM_OVERRIDES, T_ANA, AMP_THRESH, PHI_LCST,
        REG_FAILED, REG_STEADY_COLD, REG_BULK_HOPF, REG_LCST_FRONT,
        REG_GLOBAL_COLLAPSE, REG_STEADY_COLLAPSED, REG_STEADY_FRONT,
    )
    from scan_optimized import Params, simulate
    p = Params(**{**p_dict, **SIM_OVERRIDES})
    d = simulate(p)
    t = d["t"]; J = d["J"]
    idx = (t >= T_ANA[0]) & (t <= T_ANA[1])
    if idx.sum() < 50:
        regime = REG_FAILED
    else:
        Jw = J[:, idx]
        phi = p.phi_p0 / np.maximum(Jw, 1e-12)
        J_amp_max   = float((Jw.max(axis=1) - Jw.min(axis=1)).max())
        phi_max     = float(phi.max(axis=1).max())
        phi_max_min = float(phi.max(axis=1).min())
        surf_amp    = float(Jw[-1].max() - Jw[-1].min())
        is_steady = (surf_amp < AMP_THRESH and J_amp_max < AMP_THRESH)
        if is_steady:
            regime = (REG_STEADY_COLD if phi_max < PHI_LCST
                      else REG_STEADY_COLLAPSED if phi_max_min > PHI_LCST
                      else REG_STEADY_FRONT)
        else:
            regime = (REG_BULK_HOPF if phi_max < PHI_LCST
                      else REG_GLOBAL_COLLAPSE if phi_max_min > PHI_LCST
                      else REG_LCST_FRONT)
    return lbl, dict(
        t=d["t"], J_surf=d["J"][-1], x=d["x"],
        J=d["J"], theta=d["theta"],
        regime=int(regime), params=p_dict,
    )


def main():
    from fig2_data import WORKING_POINT
    from fig4_data import DATA_DIR
    cache = Path(DATA_DIR) / "fig4_repr.npz"
    points = [
        ("steady_cold",   dict(WORKING_POINT, S_chi=0.20)),
        ("lcst_front_WP", dict(WORKING_POINT)),
        ("steady_front",  dict(WORKING_POINT, Bi_T=0.18, S_chi=1.50)),
    ]
    print(f"  3 representative runs in parallel ...")
    out = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in points}
        for f in as_completed(futs):
            lbl, r = f.result()
            print(f"    {lbl} done, regime={r['regime']}")
            out[lbl] = r

    save_dict = {}
    for lbl, r in out.items():
        save_dict[f"{lbl}__t"]      = r["t"]
        save_dict[f"{lbl}__J_surf"] = r["J_surf"]
        save_dict[f"{lbl}__x"]      = r["x"]
        save_dict[f"{lbl}__J"]      = r["J"]
        save_dict[f"{lbl}__theta"]  = r["theta"]
        save_dict[f"{lbl}__regime"] = np.array(r["regime"])
        save_dict[f"{lbl}__params"] = np.array(r["params"], dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **save_dict)
    print(f"  Saved: {cache}")


if __name__ == "__main__":
    main()
