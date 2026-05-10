"""make_fig3c.py — Panel (c): time-mean J(ξ) profile, steady_cold regime."""
import os, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig3_data import panel_data_profile, draw_J_profile, PANEL_LETTERS
from fig3_style import set_style, new_panel_fig, save_panel, add_panel_label

PANEL = "c"
REGIME, _ = PANEL_LETTERS[PANEL]
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "fig3"))
CACHE = os.path.join(DATA_DIR, f"panel_{PANEL}.npz")


def compute():
    os.makedirs(DATA_DIR, exist_ok=True)
    d = panel_data_profile(REGIME)
    np.savez(CACHE, x=d["x"], J_mean=d["J_mean"],
             J_min=d["J_min"], J_max=d["J_max"], J_lcst=d["J_lcst"])
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax = new_panel_fig(with_cbar=False)
    draw_J_profile(ax, d["x"], d["J_mean"], d["J_min"],
                   d["J_max"], float(d["J_lcst"]))
    add_panel_label(ax, PANEL)
    save_panel(fig, f"panel_{PANEL}")


if __name__ == "__main__":
    render()
