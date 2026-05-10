"""
Pilot: render the slow manifold mu(J,theta) = mu_b for the working point,
overlay the PDE limit cycle from the Fig.2 cache. Visual sanity check
before promoting to a publication-style figure.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # add scripts/ to path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scan_optimized import Params, local_chem_pot, finalize_params
from fig2_data import WORKING_POINT, load_cache


def main():
    p = finalize_params(Params(**WORKING_POINT))
    print(f"mu_b = {p.m_b:.6f}")

    # --- Slow manifold via mu - mu_b = 0 contour ---
    J_grid = np.linspace(0.16, 3.0, 800)
    T_grid = np.linspace(-0.2, 5.0, 600)
    J_mesh, T_mesh = np.meshgrid(J_grid, T_grid, indexing="xy")  # rows=T, cols=J
    mu_mesh = (
        local_chem_pot(J_mesh.flatten(), T_mesh.flatten(), p)
        .reshape(J_mesh.shape)
        - p.m_b
    )

    # --- PDE limit cycle ---
    d = load_cache()
    t = d["t"]
    mask = (t >= 180) & (t <= 240)
    J_surf = d["J"][-1, mask]
    T_surf = d["theta"][-1, mask]
    J_ctr = d["J"][0, mask]
    T_ctr = d["theta"][0, mask]
    print(f"theta_surf range: [{T_surf.min():.3f}, {T_surf.max():.3f}]")
    print(f"J_surf range: [{J_surf.min():.3f}, {J_surf.max():.3f}]")
    print(f"theta_ctr range: [{T_ctr.min():.3f}, {T_ctr.max():.3f}]")
    print(f"J_ctr range: [{J_ctr.min():.3f}, {J_ctr.max():.3f}]")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(T_mesh, J_mesh, mu_mesh, levels=[0.0], colors="k", linewidths=1.5)
    # Background fill: sign of mu - mu_b (above mu_b: gel wants to collapse)
    ax.contourf(
        T_mesh, J_mesh, mu_mesh, levels=[-100, 0, 100],
        colors=["#fff8e7", "#e7f0ff"], alpha=0.5
    )

    ax.plot(T_surf, J_surf, "-", color="#d62728", lw=1.2, alpha=0.9, label="surface limit cycle")
    ax.plot(T_ctr, J_ctr, "-", color="#1f77b4", lw=1.0, alpha=0.7, label="centre")

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$J$")
    ax.set_title(r"Slow manifold $\mu(J,\theta)=\mu_b$ + PDE limit cycle")
    ax.legend(loc="upper right")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(0.1, 2.0)

    out = os.path.join(HERE, "slow_manifold_pilot.png")
    plt.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
