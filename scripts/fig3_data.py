"""
fig3_data.py — Derived ξ-resolved quantities for the Fig 3 panels.

Reuses the working-point cache produced by `fig2_data.load_cache()` and
exposes:
  * spatial envelopes (min/max/amp) for J, θ, u over the analysis window
  * time-averaged reaction-rate factors (accessibility, reactant, Arrhenius)
  * the cold-bath equilibrium  J_eq = J(θ=0, μ=μ_b),  which is the
    physically meaningful normalization for J.  J/J_eq=1 corresponds to
    the gel sitting in equilibrium with the bath at θ=0.
  * ξ_peak       = depth where J_max(ξ) is maximal
                   (mechanical-shock peak: inner boundary of the
                    propagating LCST collapse front's mechanical halo)
  * ξ_LCST       = depth where φ_max(ξ) crosses 0.5
                   (thermodynamic collapse-front locus)
  * ξ_kin        = depth where J_min(ξ) drops below 1 (kinematic
                   reference; not physically privileged but kept for
                   comparison plots).

Every per-panel script imports `panel_data()` so the heavy work runs
once and stays in sync between previews and the composite.
"""
import os
import numpy as np
from scipy.optimize import brentq

from fig2_data import load_cache, time_window, WORKING_POINT, T_START, T_END

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "Figure", "fig3"))


# ── Phase-portrait depths used by panel (c) and the composite ──────────
# Only the two spatial extremes — core (ξ=0) and surface (ξ=1) — keep the
# panel uncluttered while still showing both qualitative regimes:
# small swollen-branch loop vs full bistable-curve traversal.
PHASE_XI = (0.0, 1.0)


def equilibrium_curve(p_dict, theta_range=(0.0, 4.5), n_theta=900,
                      J_search=(0.10, 4.0), n_J=2000):
    """Trace the locus μ(J, θ) = μ_b in the (J, θ) plane.

    Returns three (theta, J) arrays — swollen, middle, collapsed — with
    NaN fillers where the corresponding root does not exist.
    """
    import sys
    sys.path.insert(0, _HERE)
    from scan_optimized import Params, finalize_params, local_chem_pot
    p = finalize_params(Params(**p_dict))
    m_b = p.m_b

    thetas = np.linspace(theta_range[0], theta_range[1], n_theta)
    Js = np.linspace(J_search[0], J_search[1], n_J)
    J_swollen   = np.full(n_theta, np.nan)
    J_middle    = np.full(n_theta, np.nan)
    J_collapsed = np.full(n_theta, np.nan)

    for k, th in enumerate(thetas):
        f = local_chem_pot(Js, np.full_like(Js, th), p) - m_b
        roots = []
        for i in range(len(Js) - 1):
            if f[i] * f[i + 1] < 0:
                try:
                    r = brentq(lambda J: float(local_chem_pot(
                        np.array([J]), np.array([th]), p)[0]) - m_b,
                        Js[i], Js[i + 1])
                    roots.append(r)
                except Exception:
                    pass
        roots = sorted(roots)
        if len(roots) == 1:
            phi = p.phi_p0 / roots[0]
            if phi < 0.5:
                J_swollen[k] = roots[0]
            else:
                J_collapsed[k] = roots[0]
        elif len(roots) >= 3:
            J_collapsed[k] = roots[0]
            J_middle[k]    = roots[1]
            J_swollen[k]   = roots[-1]
        elif len(roots) == 2:
            phi0 = p.phi_p0 / roots[0]
            phi1 = p.phi_p0 / roots[1]
            if phi0 >= 0.5 and phi1 < 0.5:
                J_collapsed[k] = roots[0]
                J_swollen[k]   = roots[1]
            else:
                J_swollen[k] = roots[-1]

    return dict(theta=thetas, swollen=J_swollen,
                middle=J_middle, collapsed=J_collapsed)


def thiele_modulus(p_dict, theta_eff=None, J_eq=None, with_Bi_c=True):
    """1D-slab Thiele modulus ζ for the swollen-branch reactant balance.

    Linearising the steady reactant equation in the swollen-branch state
    around the working point gives

        δ D₀ (1−φ_eq)^{m_diff} u'' = Da J_eq (1−φ_eq)^{m_act} T(θ_eff) · u,

    so the inverse decay length in ξ-units is

        ζ² = Da · J_eq · (1−φ_eq)^{m_act−m_diff} · T(θ_eff) / (δ · D₀),

    where T(θ) = exp[Γ_A θ / (1+ε_T θ)]. The ξ-units penetration depth
    is 1/ζ; the LCST front position should scale as ξ_LCST ≈ 1 − 1/ζ
    if ζ is the controlling dimensionless number.

    With ``with_Bi_c=True`` we add an external mass-transfer correction
    ζ_eff² = ζ² / (1 + ζ/Bi_c) so the penetration depth becomes the
    internal-controlled value when Bi_c → ∞ and the external-controlled
    value when Bi_c is small.
    """
    p = p_dict
    if J_eq is None:
        J_eq = cold_J_eq(p)
    if not np.isfinite(J_eq):
        J_eq = 1.0
    phi_eq = p["phi_p0"] / J_eq
    one_minus_phi = max(1.0 - phi_eq, 1e-12)

    if theta_eff is None:
        # Fall back to the swollen-branch 0D SS θ — only an estimate
        try:
            sys.path.insert(0, _HERE)
            from linear_stability_1d import (LSAParams, find_uniform_ss)
            lsa_keys = {f.name
                        for f in LSAParams.__dataclass_fields__.values()}
            base_lsa = {k: v for k, v in p.items() if k in lsa_keys}
            pp = LSAParams(**base_lsa)
            ss_list = find_uniform_ss(pp)
            theta_eff = ss_list[-1][2] if ss_list else 0.0
        except Exception:
            theta_eff = 0.0

    Ga = p["Gamma_A"]; eps = p["eps_T"]
    denom_T = 1.0 + eps * max(theta_eff, -1.0 / eps * 0.95)
    T_factor = float(np.exp(Ga * theta_eff / denom_T))

    diff_factor = one_minus_phi ** (p["m_act"] - p["m_diff"])
    zeta_sq = p["Da"] * J_eq * diff_factor * T_factor \
              / (p["delta"] * p["D0"])
    zeta = float(np.sqrt(max(zeta_sq, 0.0)))
    if with_Bi_c and p.get("Bi_c", 0) > 0:
        # External mass-transfer correction: ζ_eff² = ζ² / (1 + ζ/Bi_c).
        # In the small-Bi_c limit ζ_eff² → ζ·Bi_c (external control).
        zeta = zeta / np.sqrt(1.0 + zeta / p["Bi_c"])
    return zeta


def cold_J_eq(p_dict, J_search=(0.16, 4.0)):
    """Solve μ(J, θ=0) = μ_b for the cold-swollen branch J_eq.

    Returns the largest root in [J_search[0], J_search[1]] (the swollen
    branch). Falls back to NaN if no root is found.
    """
    import sys
    sys.path.insert(0, _HERE)
    from scan_optimized import Params, finalize_params, local_chem_pot
    p = finalize_params(Params(**p_dict))
    m_b = p.m_b

    def f(J):
        return float(local_chem_pot(np.array([J]),
                                    np.array([0.0]), p)[0]) - m_b

    Js = np.linspace(J_search[0], J_search[1], 800)
    fs = np.array([f(J) for J in Js])
    roots = []
    for i in range(len(Js) - 1):
        if fs[i] * fs[i + 1] < 0:
            try:
                roots.append(brentq(f, Js[i], Js[i + 1]))
            except Exception:
                pass
    if not roots:
        return float("nan")
    # Cold-swollen = largest root with phi < 0.5 (still in swollen branch)
    swollen = [r for r in roots if p.phi_p0 / r < 0.5]
    return float(max(swollen)) if swollen else float(max(roots))


def _interp_xi_first_drop(x, y, threshold):
    """Linearly interpolate the smallest ξ where y(ξ) drops below threshold."""
    above = y >= threshold
    if above.all() or not above.any():
        return float("nan")
    idx = int(np.argmax(~above))
    if idx == 0:
        return float(x[0])
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1] - threshold, y[idx] - threshold
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _interp_xi_first_rise(x, y, threshold):
    """Linearly interpolate the smallest ξ where y(ξ) crosses up through threshold."""
    above = y >= threshold
    if above.all() or not above.any():
        return float("nan")
    # First True after a False
    idx = int(np.argmax(above))
    if idx == 0:
        return float(x[0])
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1] - threshold, y[idx] - threshold
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _peak_xi(x, y):
    """Parabolic refinement of the maximum of y(x) on a discrete grid."""
    i = int(np.argmax(y))
    if 0 < i < len(x) - 1:
        ym1, y0, yp1 = y[i - 1], y[i], y[i + 1]
        denom = (ym1 - 2.0 * y0 + yp1)
        if abs(denom) > 1e-12:
            d = 0.5 * (ym1 - yp1) / denom
            return float(x[i] + d * (x[i + 1] - x[i]))
    return float(x[i])


def derived_from_arrays(x, J_w, u_w, th_w, p_dict):
    """Compute envelopes, factors and zone boundaries from time-windowed arrays."""
    p = p_dict
    phi_p0 = p["phi_p0"]
    m_act = p["m_act"]
    Gamma_A = p["Gamma_A"]
    eps_T = p["eps_T"]
    Da = p["Da"]

    phi = phi_p0 / J_w
    access = np.maximum(1.0 - phi, 1.0e-12) ** m_act
    denom = 1.0 + eps_T * np.maximum(th_w, -1.0 / eps_T * 0.95)
    arrh = np.exp(np.clip(Gamma_A * th_w / denom, -60.0, 60.0))
    R = u_w * access * arrh
    heat = Da * J_w * R

    J_min = J_w.min(axis=1)
    J_max = J_w.max(axis=1)
    th_min = th_w.min(axis=1)
    th_max = th_w.max(axis=1)
    u_min = u_w.min(axis=1)
    u_max = u_w.max(axis=1)
    phi_max_t = phi.max(axis=1)
    phi_min_t = phi.min(axis=1)

    J_eq = cold_J_eq(p)
    if not np.isfinite(J_eq):
        J_eq = 1.0

    eq_theta_max = max(4.5, float(th_w.max()) * 1.05)
    eq_curve = equilibrium_curve(p, theta_range=(0.0, eq_theta_max))

    xi_peak = _peak_xi(x, J_max)
    xi_LCST = _interp_xi_first_rise(x, phi_max_t, 0.5)
    xi_kin  = _interp_xi_first_drop(x, J_min, 1.0)

    return dict(
        x=x, J=J_w, u=u_w, theta=th_w,
        access=access, arrh=arrh, R=R, heat=heat,
        J_min=J_min, J_max=J_max,
        th_min=th_min, th_max=th_max,
        u_min=u_min, u_max=u_max,
        phi_max=phi_max_t, phi_min=phi_min_t,
        access_mean=access.mean(axis=1),
        u_mean=u_w.mean(axis=1),
        arrh_mean=arrh.mean(axis=1),
        R_mean=R.mean(axis=1),
        heat_mean=heat.mean(axis=1),
        J_eq=J_eq,
        eq_curve=eq_curve,
        xi_peak=xi_peak, xi_LCST=xi_LCST, xi_kin=xi_kin,
    )


def panel_data():
    """Return derived ξ-resolved quantities for the working-point cache."""
    d = load_cache()
    t = d["t"]
    x = d["x"]
    J = d["J"]
    u = d["u"]
    theta = d["theta"]

    idx = time_window(t)
    ts = t[idx]
    out = derived_from_arrays(
        x,
        J[:, idx],
        np.maximum(u[:, idx], 1.0e-12),
        theta[:, idx],
        WORKING_POINT,
    )
    out["t"] = ts
    out["T_START"] = T_START
    out["T_END"] = T_END
    return out


def save_panel(fig, name):
    """Save fig as PDF + PNG to Figure/fig3/<name>.{pdf,png}."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}.pdf")


# ═══════════════════════════════════════════════════════════════════
# All-regimes 3×3 mosaic (panels a–i)
# Loaders + plot helpers used by `make_fig3[a-i].py`.
# Data source: data/fig4/fig4_repr.npz (representative traces written
# by the fig4 pipeline). t-window matches the published fig3 (0, 80).
# ═══════════════════════════════════════════════════════════════════

# Row order ↔ panel-letter base.
# panels (a,b,c) = row 0 = steady_cold
# panels (d,e,f) = row 1 = lcst_front
# panels (g,h,i) = row 2 = steady_front
REGIME_NPZ_KEY = {
    "steady_cold":  "steady_cold",
    "lcst_front":   "lcst_front_WP",
    "steady_front": "steady_front",
}
REGIME_PARAMS_OVERRIDES = {
    "steady_cold":  dict(S_chi=0.20),
    "lcst_front":   dict(),
    "steady_front": dict(Bi_T=0.18, S_chi=1.50),
}
T_WINDOW_FIG3 = (0.0, 80.0)

# Panel letter → (regime_name, column) where column ∈ {"J", "theta", "profile"}.
PANEL_LETTERS = {
    "a": ("steady_cold",  "J"),
    "b": ("steady_cold",  "theta"),
    "c": ("steady_cold",  "profile"),
    "d": ("lcst_front",   "J"),
    "e": ("lcst_front",   "theta"),
    "f": ("lcst_front",   "profile"),
    "g": ("steady_front", "J"),
    "h": ("steady_front", "theta"),
    "i": ("steady_front", "profile"),
}


_FIG4_REPR = None


def _load_fig4_repr():
    """Lazy-load `data/fig4/fig4_repr.npz` (representative regime traces)."""
    global _FIG4_REPR
    if _FIG4_REPR is None:
        repr_path = os.path.normpath(os.path.join(
            _HERE, "..", "data", "fig4", "fig4_repr.npz"))
        if not os.path.exists(repr_path):
            raise FileNotFoundError(
                f"missing {repr_path} — run the fig4 pipeline first")
        _FIG4_REPR = np.load(repr_path, allow_pickle=True)
    return _FIG4_REPR


def load_regime_window(regime_name, t_window=T_WINDOW_FIG3):
    """Return (t, x, J, theta) for `regime_name` truncated to `t_window`."""
    z = _load_fig4_repr()
    key = REGIME_NPZ_KEY[regime_name]
    t = np.asarray(z[f"{key}__t"])
    x = np.asarray(z[f"{key}__x"])
    J = np.asarray(z[f"{key}__J"])
    th = np.asarray(z[f"{key}__theta"])
    mask = (t >= t_window[0]) & (t <= t_window[1])
    return dict(t=t[mask], x=x, J=J[:, mask], theta=th[:, mask])


def panel_data_kymo(regime_name, channel, t_window=T_WINDOW_FIG3):
    """Slice arrays needed for a single kymograph panel ('J' or 'theta')."""
    w = load_regime_window(regime_name, t_window)
    Z = w["J"] if channel == "J" else w["theta"]
    return dict(t=w["t"], x=w["x"], Z=Z)


def panel_data_profile(regime_name, t_window=T_WINDOW_FIG3):
    """Compute the J-profile column data: time-mean and min/max envelope.

    Following the fig3 convention (`make_fig_all_regimes.draw_J_profile`),
    the mean is taken over the **second half** of `t_window` so that
    IC-relaxation transients in the first half do not bias the steady
    attractor structure. The grey LCST line at J = phi_p0 / 0.5 marks
    where the polymer fraction crosses 0.5 (LCST collapse threshold).
    """
    w = load_regime_window(regime_name, t_window)
    t_lo, t_hi = t_window
    t_mid = 0.5 * (t_lo + t_hi)
    sub_mask = w["t"] >= t_mid
    J_sub = w["J"][:, sub_mask]
    return dict(
        x=w["x"],
        J_mean=J_sub.mean(axis=1),
        J_min=J_sub.min(axis=1),
        J_max=J_sub.max(axis=1),
        J_lcst=WORKING_POINT["phi_p0"] / 0.5,
    )


# ── Plot helpers (consume cached panel arrays) ─────────────────────

def draw_J_kymo(ax, t, x, J, fig, cax):
    """J-kymograph; LogNorm when range >½ decade, Normalize otherwise.

    The narrow steady-cold range (J ∈ [1.18, 1.30]) triggers sci-notation
    on a log colorbar (`1.30 × 10^0`) that overflows the reserved label
    margin, so we switch to a linear norm + plain decimals when J spans
    less than ~½ decade.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import FuncFormatter
    Jmin = max(0.05, float(J.min()))
    Jmax = float(J.max())
    use_log = (Jmax / Jmin) >= 3.0
    norm = LogNorm(vmin=Jmin, vmax=Jmax) if use_log else Normalize(vmin=Jmin, vmax=Jmax)
    pcm = ax.pcolormesh(t, x, J, cmap="viridis", norm=norm,
                         shading="auto", rasterized=True)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\xi$")
    cb = fig.colorbar(pcm, cax=cax)
    if not use_log:
        cb.formatter = FuncFormatter(lambda v, _: f"{v:.2f}")
        cb.update_ticks()
    # Title above the cbar instead of `set_label` (which puts a rotated
    # label to the right of the tick labels; in a tight right margin it
    # falls outside the panel and gets clipped).
    cb.ax.set_title(r"$J$", fontsize=8, pad=2)
    return pcm


def draw_theta_kymo(ax, t, x, theta, fig, cax):
    """θ-kymograph with linear Norm; colorbar drawn into `cax`."""
    from matplotlib.colors import Normalize
    th_max = max(0.5, float(theta.max()))
    pcm = ax.pcolormesh(t, x, theta, cmap="magma",
                         norm=Normalize(vmin=0.0, vmax=th_max),
                         shading="auto", rasterized=True)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\xi$")
    cb = fig.colorbar(pcm, cax=cax)
    cb.ax.set_title(r"$\theta$", fontsize=8, pad=2)
    return pcm


def draw_J_profile(ax, x, J_mean, J_min, J_max, J_lcst):
    """Time-mean J(ξ) with min/max envelope and LCST threshold line."""
    if (J_max - J_min).max() > 0.05:
        ax.fill_between(x, J_min, J_max, color="#888888", alpha=0.28,
                        lw=0, label="min/max env.")
    ax.plot(x, J_mean, color="#1f4e79", lw=1.1,
            label=r"$\langle J(\xi)\rangle$")
    ax.axhline(J_lcst, color="grey", lw=0.6, ls=":",
               label=r"$\varphi=0.5$ (LCST)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$J$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best", framealpha=0.85, handlelength=1.3, borderpad=0.3)
    ax.grid(alpha=0.3, lw=0.4)
