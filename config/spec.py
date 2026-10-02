"""Orthogonal run specification for ablations.

A :class:`RunSpec` captures the independent axes of one experiment so a single
command can vary any of them and so invalid combinations fail *fast* with a
clear message instead of crashing mid-training (NaN, F1=0, junk checkpoints).

Axes
----
``init``   — model source / initialisation:
               * ``coldstart``        (``backbone=<key>``; heads from scratch)
               * ``warmstart_uni``    (``model=<repo>``; uni only, heads pretrained)
               * ``warmstart_bi2enc`` (``warmstart=<repo>``; bi / bi2enc only)
``arch``   — ``uni`` | ``bi`` | ``bi-encoder-2-encoder``
``proj_dim`` — bi-family projection dim (``hidden`` | int); ignored for uni
``loss``   — ``{"name": "bce|focal|eiou", ...}`` (term kwargs + class_weight / ls)
``data``   — data-pipeline knobs (oversample / entity_aug / diversity)
``optim``  — lr / freeze / schedule overrides
``affine`` — span-scorer experiment name/path (dot/biaffine/cosine) or None
``dtype``  — ``bf16`` | ``fp16`` | ``fp32`` | None (resolve from registry)

``validate()`` is pure (no torch / no model load) so it is unit-testable on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from config.backbones import BACKBONES

_ARCHS = ("uni", "bi", "bi-encoder-2-encoder")
_INIT_MODES = ("coldstart", "warmstart_uni", "warmstart_bi2enc")
_BI_ARCHS = ("bi", "bi-encoder-2-encoder")
_DEBERTA_FAMILY = "deberta-v2"


class RunSpecError(ValueError):
    """Raised by ``RunSpec.validate()`` for an invalid axis combination."""


@dataclass
class RunSpec:
    # init / model source
    init_mode: str = "coldstart"
    backbone: Optional[str] = None          # coldstart
    model: Optional[str] = None             # warmstart_uni
    warmstart_from: Optional[str] = None    # warmstart_bi2enc
    # architecture
    arch: str = "uni"
    proj_dim: Any = "hidden"
    # loss / data / optim
    loss: Dict[str, Any] = field(default_factory=lambda: {"name": "bce"})
    data: Dict[str, Any] = field(default_factory=dict)
    optim: Dict[str, Any] = field(default_factory=dict)
    # span scorer + precision
    affine: Optional[str] = None
    dtype: Optional[str] = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_args(cls, args) -> "RunSpec":
        """Construct from a parsed argparse namespace (train.py)."""
        warmstart = getattr(args, "warmstart_gliner2", None)
        if warmstart:
            init_mode = "warmstart_bi2enc"
        elif getattr(args, "model", None):
            init_mode = "warmstart_uni"
        else:
            init_mode = "coldstart"

        loss = _loss_from_args(args)
        return cls(
            init_mode=init_mode,
            backbone=getattr(args, "backbone", None),
            model=getattr(args, "model", None),
            warmstart_from=warmstart,
            arch=getattr(args, "arch", "uni"),
            proj_dim=getattr(args, "proj_dim", "hidden"),
            loss=loss,
            affine=getattr(args, "affine_exp", None),
            dtype=getattr(args, "dtype", None),
        )

    # --------------------------------------------------------------- helpers
    def backbone_family(self) -> Optional[str]:
        if self.backbone and self.backbone in BACKBONES:
            return BACKBONES[self.backbone].arch
        return None

    def with_(self, **overrides) -> "RunSpec":
        """Return a copy with axes overridden — the ablation primitive."""
        return replace(self, **overrides)

    # -------------------------------------------------------------- validate
    def validate(self) -> List[str]:
        """Raise :class:`RunSpecError` on hard conflicts; return soft warnings.

        Hard errors (raise): unknown enum value, init↔arch mismatch, init↔source
        mismatch, cold-start DeBERTa with an explicit half-precision dtype.
        Warnings (return): warm-start + non-dot scorer (cold scorer), proj_dim on
        uni.
        """
        warnings: List[str] = []

        # ── enum sanity ────────────────────────────────────────────────────
        if self.arch not in _ARCHS:
            raise RunSpecError(f"unknown arch '{self.arch}'; choices: {_ARCHS}")
        if self.init_mode not in _INIT_MODES:
            raise RunSpecError(f"unknown init_mode '{self.init_mode}'; choices: {_INIT_MODES}")
        if self.loss.get("name") and self.loss["name"] not in ("bce", "focal", "eiou", "focal_eiou", "margin", "set"):
            raise RunSpecError(f"unknown loss '{self.loss['name']}'; choices: bce|focal|eiou|focal_eiou|margin|set")
        if self.dtype is not None and self.dtype not in ("bf16", "fp16", "fp32"):
            raise RunSpecError(f"unknown dtype '{self.dtype}'; choices: bf16|fp16|fp32")

        # ── init mode ↔ source field ──────────────────────────────────────
        if self.init_mode == "coldstart":
            if not self.backbone:
                raise RunSpecError("coldstart requires backbone=<key>")
            if self.model or self.warmstart_from:
                raise RunSpecError("coldstart must NOT set model/warmstart_from")
            if self.backbone not in BACKBONES:
                raise RunSpecError(f"unknown backbone '{self.backbone}'; choices: {sorted(BACKBONES)}")
        elif self.init_mode == "warmstart_uni":
            if not self.model:
                raise RunSpecError("warmstart_uni requires model=<repo>")
            if self.backbone or self.warmstart_from:
                raise RunSpecError("warmstart_uni must NOT set backbone/warmstart_from")
        else:  # warmstart_bi2enc
            if not self.warmstart_from:
                raise RunSpecError("warmstart_bi2enc requires warmstart_from=<repo>")
            if self.backbone or self.model:
                raise RunSpecError("warmstart_bi2enc must NOT set backbone/model")

        # ── init mode ↔ arch ──────────────────────────────────────────────
        if self.init_mode == "warmstart_uni" and self.arch != "uni":
            raise RunSpecError("warmstart_uni works only with --arch uni")
        if self.init_mode == "warmstart_bi2enc" and self.arch not in _BI_ARCHS:
            raise RunSpecError("warmstart_bi2enc works only with --arch bi / bi-encoder-2-encoder")
        if self.arch in _BI_ARCHS and self.init_mode == "coldstart" and not self.backbone:
            raise RunSpecError(f"--arch {self.arch} (cold-start) requires --backbone")

        # ── dtype guard: cold-start DeBERTa is unstable in half precision ──
        if (
            self.init_mode == "coldstart"
            and self.backbone_family() == _DEBERTA_FAMILY
            and self.dtype in ("bf16", "fp16")
        ):
            raise RunSpecError(
                f"cold-start DeBERTa ({self.backbone}) + dtype={self.dtype} → loss NaN "
                "(disentangled attention is unstable in half precision). Use --dtype fp32 "
                "or warm-start instead."
            )

        # ── soft warnings ─────────────────────────────────────────────────
        if self.arch == "uni" and str(self.proj_dim).lower() != "hidden":
            warnings.append(f"proj_dim={self.proj_dim} is ignored for --arch uni")
        if (
            self.init_mode in ("warmstart_uni", "warmstart_bi2enc")
            and self.affine
            and "dot" not in str(self.affine).lower()
        ):
            warnings.append(
                f"warm-start + affine '{self.affine}': the new span scorer is randomly "
                "initialised (cold) while heads are warm — consider a freeze-tier / warmup "
                "(e.g. exp08_..._freeze_scorer)."
            )

        return warnings


# ---------------------------------------------------------------------------
# Loss extraction from CLI args (orthogonal --loss-* flags)
# ---------------------------------------------------------------------------

def _loss_from_args(args) -> Dict[str, Any]:
    """Build the loss sub-spec from --loss / --loss-* flags (if present)."""
    name = getattr(args, "loss", None) or "bce"
    spec: Dict[str, Any] = {"name": name}
    gamma = getattr(args, "loss_gamma", None)
    alpha = getattr(args, "loss_alpha", None)
    if name == "focal":
        spec["gamma"] = 2.0 if gamma is None else float(gamma)
        spec["alpha"] = 0.25 if alpha is None else alpha
    elif name == "eiou":
        spec["alpha"] = 0.25 if alpha is None else float(alpha)
    elif name == "margin":
        margin = getattr(args, "loss_margin", None)
        spec["margin"] = 0.15 if margin is None else float(margin)
    elif name == "set":
        m_s = getattr(args, "set_margin", None)
        l_s = getattr(args, "set_lambda", None)
        l_f = getattr(args, "set_lambda_fp", None)
        l_l = getattr(args, "set_lambda_label", None)
        k = getattr(args, "set_topk", None)
        spec["margin_set"] = 0.5 if m_s is None else float(m_s)
        spec["lambda_set"] = 1.0 if l_s is None else float(l_s)
        spec["lambda_fp"] = 0.25 if l_f is None else float(l_f)
        spec["lambda_label"] = 0.25 if l_l is None else float(l_l)
        spec["set_topk"] = 0 if k is None else int(k)   # 0 = pool is {s > 0}
        bm = getattr(args, "set_base_margin", None)
        spec["base_margin"] = 0.0 if bm is None else float(bm)
    pos_weight = getattr(args, "pos_weight", None)
    if pos_weight is not None:
        spec["pos_weight"] = float(pos_weight)
    if getattr(args, "loss_class_weight", False):
        spec["class_weight"] = True
    ls = getattr(args, "loss_label_smoothing", None)
    if ls:
        spec["label_smoothing"] = float(ls)
    return spec
