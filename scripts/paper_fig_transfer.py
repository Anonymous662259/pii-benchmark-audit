# -*- coding: utf-8 -*-
"""Figure 1 of the PII audit paper: the controlled transfer matrix as a heatmap.

Each cell is the rare-half macro-F1 of GLiNER2 trained with objective (b) on one corpus (column),
on the test split of one corpus (row), MINUS the same score of GLiNER2 Multi-v1 without training.
The training corpus of each column is outlined. Reads out_pii_baselines/pii_transfer_matrix.json.

    python scripts/paper_fig_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SRC = Path("out_pii_baselines/pii_transfer_matrix.json")
OUT = Path("paper/figs/paper/fig_transfer.pdf")
HOMES = ["privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k", "gretel_finance", "tab"]
ROWS = HOMES + ["mapa"]
NAME = {"privy": "Privy", "pii_500k": "OpenPII", "kiji": "Kiji", "gretel": "Gretel-PII",
        "nemotron_pii": "Nemotron", "ai4p_300k": "AI4P-300k", "ai4p_400k": "AI4P-400k",
        "gretel_finance": "Gretel-Fin.", "tab": "TAB", "mapa": "MAPA"}
BASE = "GLiNER2 Multi-v1"
# diverging: warm pole (loss) -- neutral grey midpoint -- cool pole (gain)
CMAP = LinearSegmentedColormap.from_list("div", ["#b35806", "#f1a340", "#f2f2f2", "#998ec3", "#542788"])


def main() -> None:
    M = json.loads(SRC.read_text(encoding="utf-8"))
    tab = M.get("tab_control", {}).get("rare_half", {})
    diff = np.full((len(ROWS), len(HOMES)), np.nan)
    for i, ds in enumerate(ROWS):
        rec = M["corpora"][ds]["rare_half"]
        for j, h in enumerate(HOMES):
            x = tab.get(ds) if h == "tab" else rec.get(f"(b) {h}")
            if x is not None:
                diff[i, j] = x - rec[BASE]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    lim = np.nanmax(np.abs(diff))
    im = ax.imshow(diff, cmap=CMAP, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    for i in range(len(ROWS)):
        for j in range(len(HOMES)):
            v = diff[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, "0" if round(v) == 0 else f"{v:+.0f}", ha="center", va="center", fontsize=5.6,
                    color="white" if abs(v) > 0.55 * lim else "#1a1a1a")
    for j, h in enumerate(HOMES):                       # outline the training corpus of each column
        i = ROWS.index(h)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.1, ec="black"))
    ax.set_xticks(range(len(HOMES)), [NAME[h] for h in HOMES], rotation=50, ha="right")
    ax.set_yticks(range(len(ROWS)), [NAME[r] for r in ROWS])
    ax.set_xlabel("Training corpus", labelpad=2)
    ax.set_ylabel("Test corpus", labelpad=2)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("Rare-half F1 minus untrained", fontsize=6.5)
    cb.outline.set_visible(False)
    fig.tight_layout(pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {OUT}  (|diff| max {lim:.1f})")


if __name__ == "__main__":
    main()
