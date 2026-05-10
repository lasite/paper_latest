"""
style_pub.py — Minimal publication style module for figure generation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Constants ──
PRE_DOUBLE = 6.875  # inches, APS double-column width

# ── Color palette ──
C = [
    "#1f77b4",  # 0 blue
    "#d62728",  # 1 red
    "#2ca02c",  # 2 green
    "#ff7f0e",  # 3 orange
    "#9467bd",  # 4 purple
    "#8c564b",  # 5 brown
    "#e377c2",  # 6 pink
    "#7f7f7f",  # 7 gray
    "#bcbd22",  # 8 olive
    "#17becf",  # 9 cyan
]
COLORS = C  # alias

# ── Output directory ──
_OUT_DIR = Path(__file__).parent / "figures_pub"
OUT = _OUT_DIR  # alias used by some scripts


def set_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        # APS production requires Type 42 (TrueType) embedded fonts;
        # matplotlib's default Type 3 fails downstream tooling.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_panel_label(ax, label, x=None, y=None, outside=True,
                    bbox=None, fontsize=11, **kw):
    """Place '(a)' style panel label.

    outside=True  (default): just above the axes upper-left corner, no bbox.
    outside=False:           inside upper-left of axes with white bbox.
    """
    if outside:
        if x is None: x = -0.05
        if y is None: y = 1.04
        if bbox is None:
            bbox = dict(facecolor="none", edgecolor="none", pad=0)
        va, ha = "bottom", "left"
    else:
        if x is None: x = 0.01
        if y is None: y = 0.98
        if bbox is None:
            bbox = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1)
        va, ha = "top", "left"
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va=va, ha=ha,
            bbox=bbox, **kw)


def save(fig, name):
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_pdf = _OUT_DIR / f"{name}.pdf"
    path_png = _OUT_DIR / f"{name}.png"
    fig.savefig(path_pdf, dpi=600, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_pdf}")


def fig_panels(nrows, ncols, figsize=None, **kw):
    if figsize is None:
        figsize = (PRE_DOUBLE, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    return fig, axes


def kymo_show(ax, data, y_arr, t_arr, cmap="RdBu_r", label="",
              vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanpercentile(data, 2)
    if vmax is None:
        vmax = np.nanpercentile(data, 98)
    extent = [t_arr[0], t_arr[-1], y_arr[0], y_arr[-1]]
    im = ax.imshow(data, origin="lower", aspect="auto", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    if label:
        cb.set_label(label, fontsize=7)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$x/H_0$")
    return im
