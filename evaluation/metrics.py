"""Hungarian-matching evaluator for PII span prediction.

Matching criteria (both must hold):
- Same label
- IoU >= iou_threshold  OR  fuzzy match (value substring + start within fuzzy_max_dist chars)

Pass fuzzy_max_dist=None to disable the fuzzy fallback entirely (pure IoU
matching) — needed for exact-match scoring: fuzzy_max_dist=0 alone is NOT
exact-match, since the substring check still lets a truncated/extended
prediction (different end, same start) match the gold value.
"""
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from scipy.optimize import linear_sum_assignment


class HungarianEvaluator:
    name = "hung"

    def __init__(self, iou_threshold: float = 0.5, fuzzy_max_dist: Optional[int] = 3):
        self.iou_threshold = iou_threshold
        self.fuzzy_max_dist = fuzzy_max_dist

    def _iou(self, a, b) -> float:
        inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
        union = max(a["end"], b["end"]) - min(a["start"], b["start"])
        return inter / union if union > 0 else 0.0

    def _fuzzy_match(self, pred, gt) -> bool:
        if self.fuzzy_max_dist is None:
            return False
        pv = str(pred.get("value", pred.get("text", ""))).strip()
        gv = str(gt.get("value", gt.get("text", ""))).strip()
        if not pv or not gv:
            return False
        substring_ok = pv in gv or gv in pv
        start_ok = abs(pred.get("start", 0) - gt.get("start", 0)) <= self.fuzzy_max_dist
        return substring_ok and start_ok

    def _is_match(self, pred, gt) -> bool:
        if pred["label"] != gt["tag"]:
            return False
        return self._iou(pred, gt) >= self.iou_threshold or self._fuzzy_match(pred, gt)

    def count_example(self, gt_ents, pred_ents):
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        support = defaultdict(int)

        by_label_gt: Dict[str, list] = defaultdict(list)
        by_label_pred: Dict[str, list] = defaultdict(list)
        for g in gt_ents:
            by_label_gt[g["tag"]].append(g)
            support[g["tag"]] += 1
        for p in pred_ents:
            by_label_pred[p["label"]].append(p)

        all_labels = set(by_label_gt) | set(by_label_pred)
        for label in all_labels:
            gts = by_label_gt[label]
            preds = by_label_pred[label]
            if not gts:
                fp[label] += len(preds)
                continue
            if not preds:
                fn[label] += len(gts)
                continue
            # Build cost matrix (1 = no match, 0 = match)
            cost = [[0 if self._is_match(p, g) else 1 for g in gts] for p in preds]
            row_idx, col_idx = linear_sum_assignment(cost)
            matched = sum(1 for r, c in zip(row_idx, col_idx) if cost[r][c] == 0)
            tp[label] += matched
            fp[label] += len(preds) - matched
            fn[label] += len(gts) - matched

        return tp, fp, fn, support


def compute_metrics_df(
    total_tp: Dict[str, int],
    total_fp: Dict[str, int],
    total_fn: Dict[str, int],
    total_sup: Dict[str, int],
) -> pd.DataFrame:
    """Compute per-label P/R/F1 from aggregated TP/FP/FN counts."""
    rows = []
    for tag in sorted(set(total_tp) | set(total_fp) | set(total_fn)):
        tp = total_tp.get(tag, 0)
        fp = total_fp.get(tag, 0)
        fn = total_fn.get(tag, 0)
        sup = total_sup.get(tag, 0)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"label": tag, "precision": p, "recall": r, "f1": f1,
                     "support": sup, "tp": tp, "fp": fp, "fn": fn})
    if not rows:
        return pd.DataFrame(columns=["precision", "recall", "f1", "support", "tp", "fp", "fn"],
                            index=pd.Index([], name="label"))
    return pd.DataFrame(rows).set_index("label")
