#!/usr/bin/env python3
"""
diag_phase_timescales.py — B1: per-phase timescale decomposition of
the LCST-front cycle at the working point.

Reads the cached working-point trajectory and for each of the four
canonical cycle phases reports:
  * phase duration (Delta t)
  * mean rate of change of J and theta during the phase
  * fraction of cycle time
producing the numbers that go into the §IV.B slow-vs-fast claim.

Output: data/fig5/phase_timescales_WP.json (machine-readable)
        + stdout summary.

Uses the same find_peaks-based detector as make_fig2_mechanism.py /
make_slow_manifold.py for consistency.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fig2_data import load_cache, T_START, T_END

OUT_DIR = _HERE.parent / "data" / "fig5"


def detect_one_cycle(theta_t, J_t, t_t):
    """Identify one full cycle bracketed by two consecutive theta peaks."""
    Ts = np.asarray(theta_t)
    Js = np.asarray(J_t)
    ts = np.asarray(t_t)
    peaks, _ = find_peaks(Ts, prominence=0.5, distance=30)
    troughs, _ = find_peaks(-Ts, prominence=0.5, distance=30)
    if len(peaks) < 2 or len(troughs) < 1:
        raise RuntimeError("could not detect peaks/troughs")
    # Walk peaks forward until we find one with a preceding and a
    # following trough plus a following peak (so we can bracket a full
    # cycle).
    p0 = None
    p1 = None
    t_pre = None
    for k in range(len(peaks) - 1):
        cand = peaks[k]
        nxt  = peaks[k + 1]
        pre  = troughs[troughs < cand]
        if len(pre) == 0:
            continue
        between = troughs[(troughs > cand) & (troughs < nxt)]
        if len(between) == 0:
            continue
        p0    = cand
        p1    = nxt
        t_pre = pre[-1]
        break
    if p0 is None:
        raise RuntimeError("could not bracket a full cycle")
    return t_pre, p0, p1, ts, Js, Ts


def split_phases(t_pre, p0, p1, ts, Js, Ts):
    """Return per-phase index ranges [a, b) for ignite/collapse/cool/swell."""
    # Within [t_pre, p0]: rising flank (ignition). Identify maximum dT/dt.
    # Phase 1 = t_pre -> p0  (entire rising flank for theta, J on swollen branch)
    # Phase 2 = p0 -> trough_after_p0 (J fast collapse)
    # Phase 3 = trough_after_p0 -> next peak start (theta cooling on collapsed)
    # Phase 4 = (re-swell) — actually within phase 3 the J jumps back up;
    #           detect via max dJ/dt.
    troughs, _ = find_peaks(-Ts, prominence=0.5, distance=30)
    after_p0 = troughs[(troughs > p0) & (troughs < p1)]
    if len(after_p0) == 0:
        raise RuntimeError("no trough between consecutive peaks")
    tr1 = after_p0[0]
    # Phase 4: re-swell. Find max dJ/dt between tr1 and p1.
    dJ = np.gradient(Js, ts)
    if tr1 + 1 < p1:
        i_swell = tr1 + int(np.argmax(dJ[tr1: p1]))
    else:
        i_swell = tr1
    # Phase 2: collapse. Find min dJ/dt between p0 and tr1.
    if p0 + 1 < tr1:
        i_collapse = p0 + int(np.argmin(dJ[p0: tr1]))
    else:
        i_collapse = p0
    # Take phase 2 as a +/- 5% window around the min-dJ/dt point (within p0..tr1)
    # Actually let's define phase 2 by the J-fall: from when J starts dropping
    # past 95% of its peak to when J reaches 105% of its trough.
    # Simpler: the "fast jumps" are nearly instantaneous on the cycle; we
    # quantify them by their characteristic timescale, not exact bounds.
    return dict(
        phase1=(t_pre, p0),               # ignition (slow on swollen branch)
        phase2=(p0, tr1),                 # collapse + cooling-onset; we'll
                                          # split at i_collapse for the rate
        phase2_jump_idx=i_collapse,
        phase3=(tr1, i_swell),            # cool/quench
        phase4=(i_swell, p1),             # re-swell + ignition restart
        phase4_jump_idx=i_swell,
    )


def phase_stats(name, a, b, ts, Js, Ts):
    if b <= a + 1:
        return dict(name=name, dt=0.0, dJ=0.0, dtheta=0.0,
                    dJ_dt=np.nan, dtheta_dt=np.nan)
    dt = float(ts[b - 1] - ts[a])
    dJ = float(Js[b - 1] - Js[a])
    dtheta = float(Ts[b - 1] - Ts[a])
    return dict(
        name=name, dt=dt, dJ=dJ, dtheta=dtheta,
        dJ_dt=dJ / dt if dt > 0 else np.nan,
        dtheta_dt=dtheta / dt if dt > 0 else np.nan,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_cache()
    t = d["t"]
    mask = (t >= T_START) & (t <= T_END)
    ts = t[mask]
    Js = d["J"][-1, mask]    # surface
    Ts = d["theta"][-1, mask]

    t_pre, p0, p1, ts2, Js2, Ts2 = detect_one_cycle(Ts, Js, ts)
    period = float(ts2[p1] - ts2[t_pre])
    print(f"  Working point cycle: t={ts2[t_pre]:.2f} -> {ts2[p1]:.2f}, "
          f"period={period:.3f}", flush=True)

    bounds = split_phases(t_pre, p0, p1, ts2, Js2, Ts2)
    rows = [
        phase_stats("phase1: ignite (slow, swollen branch)",
                    bounds["phase1"][0], bounds["phase1"][1], ts2, Js2, Ts2),
        # phase 2 split into the slow approach + the actual fast jump
        phase_stats("phase2 window (peak->trough_theta)",
                    bounds["phase2"][0], bounds["phase2"][1], ts2, Js2, Ts2),
        phase_stats("phase3: cool/quench (slow, collapsed branch)",
                    bounds["phase3"][0], bounds["phase3"][1], ts2, Js2, Ts2),
        phase_stats("phase4 window (re-swell + restart)",
                    bounds["phase4"][0], bounds["phase4"][1], ts2, Js2, Ts2),
    ]
    print()
    print(f"  {'phase':<55s} {'dt':>7s} {'dJ':>7s} {'dtheta':>7s} "
          f"{'dJ/dt':>7s} {'dtheta/dt':>9s}  fraction", flush=True)
    for r in rows:
        frac = r["dt"] / period if period > 0 else 0.0
        print(f"  {r['name']:<55s} {r['dt']:7.3f} {r['dJ']:7.3f} "
              f"{r['dtheta']:7.3f} {r['dJ_dt']:7.2f} "
              f"{r['dtheta_dt']:9.3f}  {frac*100:5.1f}%", flush=True)

    # Fast-jump rates: take the actual dJ/dt extrema (tight, point-wise).
    dJ_dt = np.gradient(Js2, ts2)
    # Restrict each search to the appropriate sub-window to avoid
    # confusing collapse vs re-swell.
    i_c0 = p0
    i_c1 = bounds["phase3"][0]   # tr1
    i_s0 = bounds["phase3"][0]   # tr1
    i_s1 = p1
    i_c = i_c0 + int(np.argmin(dJ_dt[i_c0: max(i_c0 + 2, i_c1)]))
    i_s = i_s0 + int(np.argmax(dJ_dt[i_s0: max(i_s0 + 2, i_s1)]))
    rate_collapse = float(dJ_dt[i_c])
    rate_reswell  = float(dJ_dt[i_s])

    print()
    print(f"  Fast-jump rates (point-wise extreme dJ/dt):", flush=True)
    print(f"    collapse  dJ/dt = {rate_collapse:8.2f}  "
          f"(at t={ts2[i_c]:.2f}, J={Js2[i_c]:.3f})", flush=True)
    print(f"    re-swell  dJ/dt = {rate_reswell:8.2f}  "
          f"(at t={ts2[i_s]:.2f}, J={Js2[i_s]:.3f})", flush=True)

    # Slow-rate proxy: phase 3 dtheta/dt (cooling)
    slow_rate_theta = rows[2]["dtheta_dt"]
    fast_rate_J = max(abs(rate_collapse), abs(rate_reswell))
    if slow_rate_theta:
        ratio_J_to_theta = abs(fast_rate_J / slow_rate_theta)
        print(f"\n  slow vs fast ratio (|fast J|/|slow theta|): "
              f"{ratio_J_to_theta:.1f}", flush=True)

    # Save machine-readable record
    summary = {
        "period": period,
        "phases": [
            {**r, "fraction_of_period": r["dt"] / period}
            for r in rows
        ],
        "fast_collapse_rate_J": rate_collapse,
        "fast_reswell_rate_J": rate_reswell,
    }
    out = OUT_DIR / "phase_timescales_WP.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
