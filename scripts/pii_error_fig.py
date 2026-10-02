#!/usr/bin/env python3
"""One figure in place of the two per-corpus PII error tables.

``tab:pii_error_low`` and ``tab:pii_error_high`` are the same six categories read
in two support buckets, so they collapse into a single two-panel figure.  Counts
are divided by the gold support of their own bucket, which is what makes the four
corpora comparable and is already how the paper's totals are quoted.

The numbers are recomputed here from the stored prediction dumps through
``scripts.error_breakdown`` -- the same functions that produced the tables -- so
the figure cannot drift away from the text.

    python scripts/pii_error_fig.py
    python scripts/pii_error_fig.py --out figs/paper --no-recompute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.error_breakdown import (PII_ROOT, PII_BCE, PII_SET, PII_SEEDS,
                                     mean_counts, rare_labels)

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.3,
    "legend.handleheight": 0.8,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.0,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0.0,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Okabe--Ito; distinguishable in greyscale and for colour-blind readers.
CORPORA = [("privy", "Privy", "#0072B2"),
           ("pii_500k", "OpenPII", "#D55E00"),
           ("kiji", "Kiji", "#009E73"),
           ("gretel", "Gretel", "#CC79A7")]

# (key in the counters, label on the axis)
CATS = [("correct", "Correct"),
        ("type", "Type"),
        ("boundary", "Boundary"),
        ("boundary_type", "Boundary+Type"),
        ("missed", "Missed"),
        ("spurious", "Spurious")]
GOLD_SIDE = ("correct", "type", "boundary", "boundary_type", "missed")

BUCKETS = [("rare/", "Low-support labels"), ("freq/", "High-support labels")]


def measure() -> dict:
    """Per-corpus, per-bucket delta counts and the gold denominator."""
    out = {}
    for key, _, _ in CORPORA:
        bce_paths = [PII_ROOT / PII_BCE / "gliner2" / key / f"seed-{s}" / "test"
                     / "predictions" / "all.jsonl" for s in PII_SEEDS]
        scl_paths = [PII_ROOT / PII_SET / "gliner2" / key / f"seed-{s}" / "test"
                     / "predictions" / "all.jsonl" for s in PII_SEEDS]
        missing = [p for p in bce_paths + scl_paths if not p.is_file()]
        if missing:
            raise SystemExit(f"{key}: missing {len(missing)} dump(s), first={missing[0]}")
        rare = rare_labels(bce_paths[0])
        bce, scl = mean_counts(bce_paths, rare), mean_counts(scl_paths, rare)
        entry = {}
        for prefix, _ in BUCKETS:
            gold = sum(bce.get(prefix + c, 0) for c in GOLD_SIDE)
            entry[prefix] = {"gold": gold,
                             "delta": {c: scl.get(prefix + c, 0) - bce.get(prefix + c, 0)
                                       for c, _ in CATS}}
        out[key] = entry
    return out


def draw(data: dict, out_dir: Path) -> dict:
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.45), sharex=True, sharey=True)

    n_series = len(CORPORA) + 1          # four corpora plus the pooled total
    slot = 0.86
    height = slot / n_series
    summary = {}

    for ax, (prefix, title) in zip(axes, BUCKETS):
        gold_total = sum(data[k][prefix]["gold"] for k, _, _ in CORPORA)
        rows = {}
        for ci, (cat, _) in enumerate(CATS):
            base = len(CATS) - 1 - ci    # first category at the top
            total_delta = sum(data[k][prefix]["delta"][cat] for k, _, _ in CORPORA)
            for si, (key, name, colour) in enumerate(CORPORA):
                d = data[key][prefix]["delta"][cat]
                g = data[key][prefix]["gold"]
                pct = 100 * d / g if g else 0.0
                y = base + slot / 2 - height * (si + 0.5)
                ax.barh(y, pct, height=height * 0.88, color=colour,
                        linewidth=0, zorder=3,
                        label=name if ci == 0 and ax is axes[0] else None)
            pct_total = 100 * total_delta / gold_total if gold_total else 0.0
            y = base + slot / 2 - height * (n_series - 0.5)
            ax.barh(y, pct_total, height=height * 0.88, color="#000000",
                    linewidth=0, zorder=3,
                    label="All four (pooled)" if ci == 0 and ax is axes[0] else None)
            # The pooled value is the number quoted in the text, so print it.
            pad = 0.055 if pct_total >= 0 else -0.055
            ax.text(pct_total + pad, y, f"{pct_total:+.2f}", va="center",
                    ha="left" if pct_total >= 0 else "right", fontsize=6.2,
                    color="#000000", zorder=4)
            rows[cat] = {"pooled_delta": total_delta, "pooled_pct": pct_total}

        ax.axvline(0.0, color="#000000", linewidth=0.6, zorder=2)
        for ci in range(len(CATS) - 1):
            ax.axhline(ci + 0.5, color="#DDDDDD", linewidth=0.5, zorder=1)
        # Spurious is keyed by the PREDICTED label, unlike the five above it.
        ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)

        ax.set_yticks(range(len(CATS)))
        ax.set_yticklabels([lab for _, lab in CATS][::-1])
        ax.set_ylim(-0.55, len(CATS) - 0.45)
        ax.set_title(f"{title}  ({gold_total:,.0f} gold spans)", pad=4)
        summary[prefix] = {"gold": gold_total, "categories": rows}

    # One shared scale: the point of the figure is that the low-support changes are
    # the larger ones relative to their own gold support, which a per-panel scale
    # would hide.
    axes[0].set_xlim(-2.75, 1.95)
    axes[0].set_xticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    fig.text(0.5, 0.012, "Change in count, SCL $-$ BCE   (% of gold spans in bucket)",
             ha="center", va="top", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.145))
    fig.subplots_adjust(wspace=0.06)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_pii_errors.{ext}")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="figs/paper")
    parser.add_argument("--cache", default="figs/paper/pii_errors.json")
    parser.add_argument("--no-recompute", action="store_true",
                        help="Draw from the cached measurement instead of the dumps.")
    args = parser.parse_args()

    cache = Path(args.cache)
    if args.no_recompute and cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))["per_corpus"]
    else:
        data = measure()

    summary = draw(data, Path(args.out))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"per_corpus": data, "pooled": summary}, indent=2),
                     encoding="utf-8")

    for prefix, title in BUCKETS:
        print(f"\n{title}  ({summary[prefix]['gold']:,.0f} gold spans)")
        for cat, label in CATS:
            row = summary[prefix]["categories"][cat]
            print(f"  {label:15s} {row['pooled_delta']:+8.1f}  ({row['pooled_pct']:+.2f}%)")
    print(f"\nwrote {args.out}/fig_pii_errors.pdf (+ .png), {args.cache}")


if __name__ == "__main__":
    main()
