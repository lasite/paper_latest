"""make_fig3a.py — Panel (a): J kymograph, steady_cold regime."""
import os, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fig3_data import panel_data_kymo, draw_J_kymo, PANEL_LETTERS
from fig3_style import set_style, new_panel_fig, save_panel, add_panel_label

PANEL = "a"
REGIME, CHANNEL = PANEL_LETTERS[PANEL]
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data", "fig3"))
CACHE = os.path.join(DATA_DIR, f"panel_{PANEL}.npz")


def compute():
    os.makedirs(DATA_DIR, exist_ok=True)
    d = panel_data_kymo(REGIME, CHANNEL)
    np.savez(CACHE, t=d["t"], x=d["x"], Z=d["Z"])
    print(f"  Saved cache: {CACHE}")


def render():
    if not os.path.exists(CACHE):
        compute()
    d = np.load(CACHE)
    set_style()
    fig, ax, cax = new_panel_fig(with_cbar=True)
    draw_J_kymo(ax, d["t"], d["x"], d["Z"], fig, cax)
    add_panel_label(ax, PANEL)
    save_panel(fig, f"panel_{PANEL}")


if __name__ == "__main__":
    render()
