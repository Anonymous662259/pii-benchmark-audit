# -*- coding: utf-8 -*-
"""Every number in the results tables of acl2023_rare_pii.tex, from the prediction dumps.

PROTOCOL.md round, arms (a) bce, (b) bce_bundle, (c) set on four PII corpora.  The
script writes one LaTeX fragment per table into paper/generated/, which
the paper \\input's, plus out_pii_baselines/rare_pii_tables.json.  A missing arm or seed
turns its cells into "--", so the tables fill in as runs land -- nothing is typed by hand.

Batch 1 (PROTOCOL.md section 10): seeds 42 and 1337 only, and the tuned threshold is
searched on {0.50, 0.55, ..., 0.90} because the pilot dumps of (a) and (c) keep only
spans scored >= 0.5.  A seed counts for an arm only once it exists on all four corpora.

    python scripts/rare_pii_tables.py                  # 5000 bootstrap resamples
    python scripts/rare_pii_tables.py --resamples 500  # quick look
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pii_significance import metrics_from, summarise  # noqa: E402
from error_breakdown import classify_sentence  # noqa: E402
from cross_gold_coverage import sentence as coverage_sentence  # noqa: E402
from evaluation.metrics import HungarianEvaluator  # noqa: E402

STRICT = HungarianEvaluator(iou_threshold=1.0, fuzzy_max_dist=None)
ROOT = Path("out_pii_baselines")
ARMS = {"a": "v1-bce-baseline-50k", "b": "v1-bce-bundle-50k", "c": "v1-set-consistent-50k"}
ARM_NAME = {"a": "(a) BCE", "b": "(b) Class-aware", "c": "(c) + Competition terms"}
DATASETS = ["privy", "pii_500k", "kiji", "gretel"]          # paper column order
CANDIDATE_SEEDS = [42, 1337, 2024]
GRID = [round(0.50 + 0.05 * i, 2) for i in range(9)]       # batch-1 threshold grid
FIXED = 0.5
BUCKETS = ("rare", "freq")
GOLD_CATS = ("correct", "type", "boundary", "boundary_type", "missed")
OUT_DIR = Path("paper/generated")
NA = r"\na"


# ----------------------------------------------------------------------------- loading

def dump(arm: str, ds: str, seed: int, split: str) -> Path:
    return ROOT / ARMS[arm] / "gliner2" / ds / f"seed-{seed}" / split / "predictions" / "all.jsonl"


@lru_cache(maxsize=None)
def seeds_of(arm: str) -> Tuple[int, ...]:
    """Seeds of this arm that exist on EVERY corpus (PROTOCOL.md section 10, batch 2).

    A seed that has landed for only some corpora is left out entirely, so an unfinished
    or unaffordable corpus can never turn into a per-corpus choice of seeds.
    """
    return tuple(s for s in CANDIDATE_SEEDS
                 if all(dump(arm, ds, s, sp).is_file() for ds in DATASETS for sp in ("validation", "test")))


def has_arm(arm: str, ds: str) -> bool:
    return bool(seeds_of(arm))


@lru_cache(maxsize=None)
def rows(path: Path) -> Tuple[Tuple[tuple, tuple], ...]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out.append((tuple(map(_freeze, r.get("gt") or [])),
                            tuple(map(_freeze, r.get("predictions") or []))))
    return tuple(out)


def _freeze(d: dict) -> tuple:
    return tuple(sorted(d.items()))


def _thaw(t: tuple) -> dict:
    return dict(t)


def filtered(path: Path, t: float) -> List[Tuple[List[dict], List[dict]]]:
    return [([_thaw(g) for g in gt], [_thaw(p) for p in preds if dict(p).get("score", 1.0) >= t])
            for gt, preds in rows(path)]


@lru_cache(maxsize=None)
def labels_of(ds: str, split: str) -> Tuple[str, ...]:
    """Every label seen in gold or predictions of any available arm/seed for this split."""
    seen = set()
    for arm in ARMS:
        for s in seeds_of(arm):
            p = dump(arm, ds, s, split)
            if p.is_file():
                for gt, preds in rows(p):
                    seen |= {dict(g)["tag"] for g in gt} | {dict(x)["label"] for x in preds}
    return tuple(sorted(seen))


@lru_cache(maxsize=None)
def counts(path: Path, t: float, labels: Tuple[str, ...]) -> np.ndarray:
    """(n_sentences, n_labels, 4) tp/fp/fn/support under strict exact match, spans >= t."""
    index = {l: i for i, l in enumerate(labels)}
    out = []
    for gt, preds in filtered(path, t):
        tp, fp, fn, sup = STRICT.count_example(gt, preds)
        cell = np.zeros((len(labels), 4))
        for src, col in ((tp, 0), (fp, 1), (fn, 2), (sup, 3)):
            for l, v in src.items():
                cell[index[l], col] += v
        out.append(cell)
    return np.asarray(out)


@lru_cache(maxsize=None)
def rare_mask(ds: str) -> np.ndarray:
    """Lower half of the labels by test gold support -- fixed once per corpus."""
    labels = labels_of(ds, "test")
    ref = next(dump(a, ds, seeds_of(a)[0], "test") for a in ARMS if seeds_of(a))
    support = counts(ref, FIXED, labels)[:, :, 3].sum(axis=0)
    present = [i for i in range(len(labels)) if support[i] > 0]
    order = sorted(present, key=lambda i: support[i])
    mask = np.zeros(len(labels), dtype=bool)
    mask[order[:max(1, math.ceil(len(present) / 2))]] = True
    return mask


def rare_set(ds: str) -> set:
    return {l for l, m in zip(labels_of(ds, "test"), rare_mask(ds)) if m}


# ----------------------------------------------------------------------------- metrics

def score(arm: str, ds: str, seed: int, split: str, t: float) -> Tuple[float, float, float]:
    labels = labels_of(ds, split)
    mask = rare_mask(ds) if split == "test" else np.zeros(len(labels), dtype=bool)
    return metrics_from(counts(dump(arm, ds, seed, split), t, labels).sum(axis=0), mask)


@lru_cache(maxsize=None)
def tuned_t(arm: str, ds: str, seed: int) -> float:
    """Threshold on GRID with the best validation macro-F1 (ties -> lowest threshold)."""
    return max(GRID, key=lambda t: (score(arm, ds, seed, "validation", t)[0], -t))


def run_metrics(arm: str, ds: str, mode: str) -> Optional[List[Tuple[float, float, float]]]:
    if not has_arm(arm, ds):
        return None
    return [score(arm, ds, s, "test", FIXED if mode == "fixed" else tuned_t(arm, ds, s))
            for s in seeds_of(arm)]


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def fmt(m: float, sd: Optional[float] = None) -> str:
    return f"${m:.2f}$" if sd is None else f"${m:.2f}_{{\\pm {sd:.2f}}}$"


def signed(x: float) -> str:
    return f"${x:+.2f}$"


# ----------------------------------------------------------------------------- tables

def table_main(result: Dict) -> str:
    lines = []
    for mode, title in (("fixed", r"Fixed ($p_0=0.5$)"), ("tuned", "Tuned on validation")):
        for k, arm in enumerate(ARMS):
            cells = [title if k == 0 else "", ARM_NAME[arm]]
            per_ds = {ds: run_metrics(arm, ds, mode) for ds in DATASETS}
            for ds in DATASETS:
                m = per_ds[ds]
                cells.append(NA if m is None else fmt(*mean_sd([x[1] for x in m])))
            complete = all(per_ds[ds] is not None for ds in DATASETS)
            for j in (1, 0, 2):   # rare-half avg, macro avg, micro avg
                if complete:
                    avg = statistics.mean(mean_sd([x[j] for x in per_ds[ds]])[0] for ds in DATASETS)
                    cells.append(fmt(avg))
                    result.setdefault("main", {}).setdefault(mode, {}).setdefault(arm, {})[
                        ("macro", "rare", "micro")[j] + "_avg"] = round(avg, 3)
                else:
                    cells.append(NA)
            for ds in DATASETS:
                if per_ds[ds] is not None:
                    result.setdefault("per_corpus", {}).setdefault(mode, {}).setdefault(arm, {})[ds] = {
                        name: [round(x[j], 3) for x in per_ds[ds]]
                        for j, name in enumerate(("macro", "rare", "micro"))}
            lines.append(" & ".join(cells) + r" \\")
        if mode == "fixed":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def table_per_corpus() -> str:
    lines = []
    for mode, title in (("fixed", "Fixed"), ("tuned", "Tuned")):
        for k, arm in enumerate(ARMS):
            cells = [title if k == 0 else "", f"({arm})"]
            for j in (0, 2):
                for ds in DATASETS:
                    m = run_metrics(arm, ds, mode)
                    cells.append(NA if m is None else fmt(*mean_sd([x[j] for x in m])))
            lines.append(" & ".join(cells) + r" \\")
        if mode == "fixed":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def bootstrap_pair(ds: str, x: str, y: str, resamples: int, rng: np.random.Generator):
    """Draws of metric(y) - metric(x), seeds averaged, sentences shared, tuned thresholds."""
    labels = labels_of(ds, "test")
    mask = rare_mask(ds)
    sx, sy = seeds_of(x), seeds_of(y)       # may differ: each arm averages its own seeds
    A = np.stack([counts(dump(x, ds, s, "test"), tuned_t(x, ds, s), labels) for s in sx])
    B = np.stack([counts(dump(y, ds, s, "test"), tuned_t(y, ds, s), labels) for s in sy])
    n = A.shape[1]
    obs = (np.mean([metrics_from(B[i].sum(0), mask) for i in range(len(sy))], axis=0)
           - np.mean([metrics_from(A[i].sum(0), mask) for i in range(len(sx))], axis=0))
    fa, fb = A.reshape(len(sx), n, -1), B.reshape(len(sy), n, -1)
    draws = np.empty((resamples, 3))
    done = 0
    while done < resamples:
        size = min(250, resamples - done)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(np.float64)
        ta, tb = np.einsum("bn,snf->bsf", w, fa), np.einsum("bn,snf->bsf", w, fb)
        for k in range(size):
            da = np.mean([metrics_from(ta[k, i].reshape(-1, 4), mask) for i in range(len(sx))], axis=0)
            db = np.mean([metrics_from(tb[k, i].reshape(-1, 4), mask) for i in range(len(sy))], axis=0)
            draws[done + k] = db - da
        done += size
    return obs, draws


def per_label_f1(arm: str, ds: str, seed: int) -> np.ndarray:
    """Per-label F1 (%) on test at this run's tuned threshold, in labels_of(ds, 'test') order."""
    labels = labels_of(ds, "test")
    tot = counts(dump(arm, ds, seed, "test"), tuned_t(arm, ds, seed), labels).sum(0)
    tp, fp, fn = tot[:, 0], tot[:, 1], tot[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        return 100 * np.where(p + r > 0, 2 * p * r / (p + r), 0.0)


def sign_flip_p(d: Sequence[float]) -> float:
    """Exact two-sided sign-flip permutation p for mean(d) != 0 (all 2^n sign patterns)."""
    d = np.asarray(d, dtype=float)
    signs = np.array(np.meshgrid(*[[1.0, -1.0]] * len(d))).reshape(len(d), -1).T   # (2^n, n)
    null = np.abs(signs @ d) / len(d)
    return float((null >= abs(d.mean()) - 1e-12).mean())


def table_tests(result: Dict) -> str:
    """Seed-level sign-flip test and label-level Wilcoxon for each contrast, tuned threshold.

    Seed level: one difference per (corpus, seed) over the seeds both arms share, so the
    unit is a training run and seed-to-seed variation is inside the test.
    Label level: per-label F1 averaged over those seeds, one difference per rare label,
    pooled over the four corpora -- whether rare types move together or a few carry the mean.
    """
    from scipy.stats import wilcoxon

    lines = []
    for x, y in (("a", "b"), ("b", "c"), ("a", "c")):
        label = f"({y})$-$({x})"
        seeds = [s for s in seeds_of(x) if s in seeds_of(y)]
        if not seeds:
            lines.append(" & ".join([label] + [NA] * 7) + r" \\")
            continue
        diffs = {m: [] for m in ("macro", "rare", "micro")}
        label_d = []
        for ds in DATASETS:
            for s in seeds:
                mx, my = score(x, ds, s, "test", tuned_t(x, ds, s)), score(y, ds, s, "test", tuned_t(y, ds, s))
                for j, m in enumerate(("macro", "rare", "micro")):
                    diffs[m].append(my[j] - mx[j])
            fx = np.mean([per_label_f1(x, ds, s) for s in seeds], axis=0)
            fy = np.mean([per_label_f1(y, ds, s) for s in seeds], axis=0)
            label_d.extend((fy - fx)[rare_mask(ds)].tolist())
        label_d = np.asarray(label_d)
        nz = label_d[np.abs(label_d) > 1e-9]
        p_label = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
        out = {m: {"diff": round(float(np.mean(v)), 3), "p_sign_flip": round(sign_flip_p(v), 4),
                   "n_pairs": len(v), "n_positive": int(sum(dd > 0 for dd in v))}
               for m, v in diffs.items()}
        out["labels"] = {"n": int(len(label_d)), "improved": int((label_d > 1e-9).sum()),
                         "worse": int((label_d < -1e-9).sum()),
                         "median": round(float(np.median(label_d)), 3), "p_wilcoxon": round(p_label, 4)}
        out["seeds"] = seeds
        result.setdefault("tests_tuned", {})[f"{y}-{x}"] = out
        r = out["rare"]
        fmt_p = lambda p: f"${p:.3f}$" if p >= 0.001 else "$<0.001$"
        boot = result.get("contrasts_tuned", {}).get(f"{y}-{x}", {}).get("rare")
        ci = NA if boot is None else f"[{boot['ci_lo']:+.2f}, {boot['ci_hi']:+.2f}]"
        lines.append(" & ".join([
            label, signed(r["diff"]), ci, fmt_p(r["p_sign_flip"]), f"{r['n_positive']}/{r['n_pairs']}",
            f"{out['labels']['improved']}/{out['labels']['n']}", fmt_p(p_label),
            f"{signed(out['macro']['diff'])} / {signed(out['micro']['diff'])}"]) + r" \\")
    return "\n".join(lines) + "\n"


def table_contrasts(result: Dict, resamples: int) -> str:
    lines = []
    rng = np.random.default_rng(20260927)
    for x, y in (("a", "b"), ("b", "c"), ("a", "c")):
        label = f"({y})$-$({x})"
        if not all(has_arm(x, ds) and has_arm(y, ds) for ds in DATASETS):
            lines.append(f"{label} & {NA}\\ [{NA}, {NA}] & {NA} & {NA} \\\\")
            continue
        per = [bootstrap_pair(ds, x, y, resamples, rng) for ds in DATASETS]
        obs = np.mean([o for o, _ in per], axis=0)
        draws = np.mean([d for _, d in per], axis=0)
        s = {name: summarise(draws[:, j], obs[j]) for j, name in enumerate(("macro", "rare", "micro"))}
        result.setdefault("contrasts_tuned", {})[f"{y}-{x}"] = {
            **s, "per_corpus": {ds: {name: round(float(o[j]), 3) for j, name in enumerate(("macro", "rare", "micro"))}
                                for ds, (o, _) in zip(DATASETS, per)}}
        r = s["rare"]
        lines.append(f"{label} & {signed(r['diff'])}\\ [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] "
                     f"& {signed(s['macro']['diff'])} & {signed(s['micro']['diff'])} \\\\")
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=None)
def error_counts(arm: str, ds: str) -> Optional[Dict[str, float]]:
    """Seed-mean error counts at p0 = 0.5, split by support bucket."""
    if not has_arm(arm, ds):
        return None
    rare = rare_set(ds)
    per_seed = []
    for s in seeds_of(arm):
        c: Counter = Counter()
        for gt, preds in filtered(dump(arm, ds, s, "test"), FIXED):
            c.update(classify_sentence(gt, preds, rare))
        per_seed.append(c)
    keys = set().union(*per_seed)
    return {k: statistics.mean(c.get(k, 0) for c in per_seed) for k in keys}


def pooled(arm: str) -> Optional[Dict[str, float]]:
    per = [error_counts(arm, ds) for ds in DATASETS]
    if any(p is None for p in per):
        return None
    keys = set().union(*per)
    return {k: sum(p.get(k, 0.0) for p in per) for k in keys}


def gold(c: Dict[str, float], bucket: str) -> float:
    return sum(c.get(f"{bucket}/{cat}", 0.0) for cat in GOLD_CATS)


ROW_NAMES = (("correct", "Correct"), ("type", "Type error"), ("boundary", "Boundary error"),
             ("boundary_type", r"Boundary $+$ type"), ("missed", "Missed"))


def table_bce_errors(result: Dict) -> str:
    c = pooled("a")
    lines = []
    for key, name in ROW_NAMES:
        cells = [name]
        for b in BUCKETS:
            cells.append(NA if c is None else f"{100 * c.get(f'{b}/{key}', 0) / gold(c, b):.2f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = ["Spurious (per gold)"]
    for b in BUCKETS:
        cells.append(NA if c is None else f"{100 * c.get(f'{b}/spurious', 0) / gold(c, b):.2f}")
    lines.append(" & ".join(cells) + r" \\")
    if c is not None:
        result["bce_errors_pct"] = {b: {k: round(100 * c.get(f"{b}/{k}", 0) / gold(c, b), 3)
                                        for k in GOLD_CATS + ("spurious",)} for b in BUCKETS}
    return "\n".join(lines) + "\n"


def table_error_change(result: Dict) -> str:
    arms = {a: pooled(a) for a in ARMS}
    lines = []
    for key, name in ROW_NAMES + (("spurious", "Spurious"),):
        cells = [name]
        for x, y in (("a", "b"), ("b", "c")):
            for b in BUCKETS:
                if arms[x] is None or arms[y] is None:
                    cells.append(NA)
                    continue
                d = arms[y].get(f"{b}/{key}", 0) - arms[x].get(f"{b}/{key}", 0)
                pct = 100 * d / gold(arms[x], b)
                cells.append(f"${pct:+.2f}$")
                result.setdefault("error_change_pct", {}).setdefault(f"{y}-{x}", {}).setdefault(b, {})[key] = round(pct, 3)
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def train_support(ds: str) -> Counter:
    stats = json.loads((ROOT / ARMS["a"] / "pii_dataset_stats.json").read_text(encoding="utf-8"))
    return Counter(stats[ds]["labels"])


def table_rare_partition(result: Dict) -> str:
    lines = []
    for ds, name in zip(DATASETS, ("Privy", "OpenPII", "Kiji", "Gretel")):
        if not (has_arm("b", ds) and has_arm("c", ds)):
            lines.append(f"{name} & {NA} & {NA} & {NA} \\\\")
            continue
        labels = labels_of(ds, "test")
        present = [l for l, s in zip(labels, counts(dump("a", ds, seeds_of("a")[0], "test"), FIXED, labels)[:, :, 3].sum(0)) if s > 0]
        train = train_support(ds)
        by_train = sorted(present, key=lambda l: train.get(l, 0))[:max(1, math.ceil(len(present) / 2))]
        train_mask = np.array([l in set(by_train) for l in labels])
        gains = []
        for mask in (rare_mask(ds), train_mask):
            vals = {}
            for arm in ("b", "c"):
                vals[arm] = statistics.mean(
                    metrics_from(counts(dump(arm, ds, s, "test"), tuned_t(arm, ds, s), labels).sum(0), mask)[1]
                    for s in seeds_of(arm))
            gains.append(vals["c"] - vals["b"])
        overlap = 100 * len(rare_set(ds) & set(by_train)) / len(rare_set(ds))
        result.setdefault("rare_partition", {})[ds] = {"test": round(gains[0], 3), "train": round(gains[1], 3),
                                                        "overlap_pct": round(overlap, 1)}
        lines.append(f"{name} & {signed(gains[0])} & {signed(gains[1])} & {overlap:.0f}\\% \\\\")
    return "\n".join(lines) + "\n"


def table_coverage(result: Dict) -> str:
    lines = []
    for ds, name in zip(DATASETS, ("Privy", "OpenPII", "Kiji", "Gretel")):
        cells = [name]
        for arm in ARMS:
            if not has_arm(arm, ds):
                cells.append(NA)
                continue
            pct = []
            for s in seeds_of(arm):
                cross = n_gold = 0
                for gt, preds in filtered(dump(arm, ds, s, "test"), FIXED):
                    out = coverage_sentence(gt, preds, True)
                    cross += out["cross_any"]
                    n_gold += len(gt)
                pct.append(100 * cross / n_gold)
            m = statistics.mean(pct)
            result.setdefault("coverage_pct", {}).setdefault(arm, {})[ds] = round(m, 3)
            cells.append(f"{m:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


FIG_DIR = Path("paper/figs/paper")


def figure_error_change(pairs=(("a", "b"), ("b", "c")), name: str = "fig_rare_pii_errors") -> Optional[Path]:
    """Per-corpus error changes, one row per contrast, one column per support bucket.

    Styling follows scripts/pii_error_fig.py (Okabe--Ito, Times, 8 pt). The palette passes
    the dataviz validator with two warnings, relieved here by a fixed bar order that matches
    the legend, a thin dark outline on every bar, and Table tab:error_change as the table view.
    """
    if not all(has_arm(x, ds) and has_arm(y, ds) for x, y in pairs for ds in DATASETS):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pii_error_fig import RC, CORPORA, CATS, BUCKETS

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6.3, 2.2 * len(pairs)),
                             sharex=True, sharey=True, squeeze=False)
    n_series = len(CORPORA) + 1
    slot, height = 0.86, 0.86 / (len(CORPORA) + 1)
    extent = 0.0
    for r, (x, y) in enumerate(pairs):
        for c, (prefix, title) in enumerate(BUCKETS):
            ax, b = axes[r][c], prefix.rstrip("/")
            gold_total = sum(gold(error_counts(x, k), b) for k, _, _ in CORPORA)
            for ci, (cat, _) in enumerate(CATS):
                base = len(CATS) - 1 - ci
                total = 0.0
                for si, (key, label, colour) in enumerate(CORPORA):
                    cx, cy = error_counts(x, key), error_counts(y, key)
                    d = cy.get(f"{b}/{cat}", 0) - cx.get(f"{b}/{cat}", 0)
                    total += d
                    pct = 100 * d / gold(cx, b)
                    extent = max(extent, abs(pct))
                    ax.barh(base + slot / 2 - height * (si + 0.5), pct, height=height * 0.88,
                            color=colour, edgecolor="#333333", linewidth=0.3, zorder=3,
                            label=label if (ci == 0 and r == 0 and c == 0) else None)
                pooled = 100 * total / gold_total
                extent = max(extent, abs(pooled))
                yy = base + slot / 2 - height * (n_series - 0.5)
                ax.barh(yy, pooled, height=height * 0.88, color="#000000", linewidth=0, zorder=3,
                        label="All four (pooled)" if (ci == 0 and r == 0 and c == 0) else None)
                ax.text(pooled + (0.04 if pooled >= 0 else -0.04), yy, f"{pooled:+.2f}",
                        va="center", ha="left" if pooled >= 0 else "right", fontsize=6.2, zorder=4)
            ax.axvline(0.0, color="#000000", linewidth=0.6, zorder=2)
            for ci in range(len(CATS) - 1):
                ax.axhline(ci + 0.5, color="#DDDDDD", linewidth=0.5, zorder=1)
            ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
            ax.set_yticks(range(len(CATS)))
            ax.set_yticklabels([lab for _, lab in CATS][::-1])
            ax.set_ylim(-0.55, len(CATS) - 0.45)
            ax.set_title(f"({y})$-$({x}): {title.lower()}  ({gold_total:,.0f} gold spans)", pad=4)
    lim = extent * 1.25 + 0.3
    axes[0][0].set_xlim(-lim, lim)
    fig.text(0.5, 0.0, "Change in count   (% of gold spans in the bucket)", ha="center", va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.subplots_adjust(wspace=0.06, hspace=0.35)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    return FIG_DIR / f"{name}.pdf"


def _join(seeds: Sequence[int]) -> str:
    seeds = [str(s) for s in seeds]
    return seeds[0] if len(seeds) == 1 else ", ".join(seeds[:-1]) + " and " + seeds[-1]


def meta() -> str:
    """\\nSeeds / \\seedList for the text; they spell out per-arm seeds when arms differ."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    per_arm = {arm: seeds_of(arm) for arm in ARMS if seeds_of(arm)}
    counts_ = sorted({len(s) for s in per_arm.values()})
    n_text = " or ".join(words.get(n, str(n)) for n in counts_)
    if len(set(per_arm.values())) == 1:
        seed_text = _join(next(iter(per_arm.values())))
    else:
        groups: Dict[Tuple[int, ...], List[str]] = {}
        for arm, s in per_arm.items():
            groups.setdefault(s, []).append(f"({arm})")
        seed_text = "; ".join(f"{_join(s)} for {' and '.join(a)}" for s, a in groups.items())
    return (f"\\newcommand{{\\nSeeds}}{{{n_text}}}\n"
            f"\\newcommand{{\\seedList}}{{{seed_text}}}\n"
            f"\\newcommand{{\\tuneGrid}}{{$\\{{{GRID[0]:.2f},{GRID[1]:.2f},\\dots,{GRID[-1]:.2f}\\}}$}}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bootstrap", type=int, default=5000, metavar="RESAMPLES",
                    help="sentence-level paired bootstrap resamples for the test-set interval "
                         "in Table tab:contrasts (0 = skip; the decision uses the sign-flip test)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result: Dict = {"seeds": {arm: list(seeds_of(arm)) for arm in ARMS}, "grid": GRID,
                    "available": {arm: [ds for ds in DATASETS if has_arm(arm, ds)] for arm in ARMS}}
    if args.bootstrap:   # test-set interval; must run before table_tests, which prints it
        table_contrasts(result, args.bootstrap)
    tables = {
        "meta.tex": meta(),
        "tab_main_body.tex": table_main(result),
        "tab_per_corpus_body.tex": table_per_corpus(),
        "tab_tests_body.tex": table_tests(result),
        "tab_bce_errors_body.tex": table_bce_errors(result),
        "tab_error_change_body.tex": table_error_change(result),
        "tab_rare_partition_body.tex": table_rare_partition(result),
        "tab_coverage_body.tex": table_coverage(result),
    }
    for name, body in tables.items():
        (OUT_DIR / name).write_text("% generated by scripts/rare_pii_tables.py -- do not edit\n" + body,
                                    encoding="utf-8")
    fig = figure_error_change()
    result["figure"] = str(fig) if fig else None
    result["tuned_thresholds"] = {arm: {ds: [tuned_t(arm, ds, s) for s in seeds_of(arm)]
                                        for ds in DATASETS if has_arm(arm, ds)} for arm in ARMS}
    (ROOT / "rare_pii_tables.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("available:", result["available"])
    print("wrote", ", ".join(tables), "->", OUT_DIR)


if __name__ == "__main__":
    main()
