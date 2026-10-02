"""Load & apply a span-scorer (affine_adapt) experiment to a GLiNER2 model.

An experiment is a small JSON file under ``experiments/affine/configs/`` with:

    {
      "name": "exp01_biaffine_full",
      "description": "...",
      "scorer": { "score_mode": "biaffine", "biaffine_rank": 0, ... },
      "freeze": "full" | "heads" | "scorer",
      "train_overrides": { "num_epochs": 3, "encoder_lr": 1e-5 }
    }

- ``scorer``           -> passed to ``model.configure_scorer(**scorer)`` (rebuilds
                          the span scorer post-load, persists on config).
- ``freeze``           -> which param groups get ``requires_grad`` (see tiers below).
- ``train_overrides``  -> merged into TrainingConfig by the training script.

Freeze tiers (matches the affine_adapt design doc):
  - ``full``   : train everything (encoder + heads + scorer).      [best ceiling]
  - ``heads``  : freeze encoder; train heads + scorer.             [sweet spot]
  - ``scorer`` : freeze encoder + heads; train only the scorer.    [cheapest probe]
"""
import json
import os
from typing import Any, Dict

# Modules considered "task heads" (frozen in the `scorer` tier). Names that are
# absent on a given architecture are silently skipped (e.g. text_proj/schema_proj
# only exist on the bi-encoder).
_HEAD_MODULES = (
    "span_rep", "classifier", "count_pred", "count_embed",
    "text_proj", "schema_proj", "bi_classifier",
)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def load_experiment(spec: str) -> Dict[str, Any]:
    """Load an experiment by name (``exp01_biaffine_full``) or explicit path."""
    path = spec
    if not os.path.isfile(path):
        cand = os.path.join(CONFIGS_DIR, spec)
        if not cand.endswith(".json"):
            cand += ".json"
        path = cand
    if not os.path.isfile(path):
        raise FileNotFoundError(f"affine experiment not found: {spec} (looked at {path})")
    with open(path, "r", encoding="utf-8") as f:
        exp = json.load(f)
    exp.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return exp


def _set_requires_grad(module, flag: bool) -> None:
    if module is not None:
        for p in module.parameters():
            p.requires_grad = flag


def apply_freeze(model, tier: str) -> None:
    """Set requires_grad per freeze tier. The optimizer skips frozen params."""
    if tier == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif tier in ("heads", "scorer"):
        _set_requires_grad(getattr(model, "encoder", None), False)
        if tier == "scorer":
            for name in _HEAD_MODULES:
                _set_requires_grad(getattr(model, name, None), False)
        # the span scorer is always trainable in heads/scorer tiers
        _set_requires_grad(getattr(model, "span_scorer", None), True)
    else:
        raise ValueError(f"unknown freeze tier: {tier!r} (use full|heads|scorer)")


def apply_experiment(model, exp: Dict[str, Any]):
    """Configure the span scorer + apply the freeze tier. Returns the model."""
    scorer = exp.get("scorer") or {}
    if scorer:
        model.configure_scorer(**scorer)
    apply_freeze(model, exp.get("freeze", "full"))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[affine] experiment={exp.get('name')} mode={scorer.get('score_mode', 'dot')} "
        f"freeze={exp.get('freeze', 'full')} trainable={trainable:,}/{total:,} "
        f"({100 * trainable / max(total, 1):.1f}%)"
    )
    return model
