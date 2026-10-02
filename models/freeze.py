"""Utilities for freezing encoder layers to reduce catastrophic forgetting.

Used by Experiment 4 (Conservative Encoder Fine-Tuning) and any combined
strategy that wants to lock down the lower transformer layers.

Functions
---------
freeze_embeddings(model)
    Freeze token embeddings, position embeddings, and layer-norm on the
    encoder's embedding block.

freeze_encoder_layers(model, n_layers)
    Freeze the bottom ``n_layers`` transformer blocks (attention + FFN) of the
    encoder backbone.

print_frozen_summary(model)
    Print a summary of trainable vs frozen parameter counts.
"""
from __future__ import annotations

import re
from typing import Optional

import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers to navigate the model hierarchy
# ---------------------------------------------------------------------------

def _get_encoder(model: nn.Module) -> Optional[nn.Module]:
    """Return the HuggingFace encoder backbone inside a GLiNER2 / CustomExtractor."""
    # GLiNER2 exposes the backbone as .encoder (or .model.encoder)
    for attr in ("encoder", "model"):
        enc = getattr(model, attr, None)
        if enc is not None:
            return enc
    return None


def _get_transformer_layers(encoder: nn.Module):
    """Return the list of transformer blocks from an encoder."""
    for attr_path in (
        "encoder.layer",          # DeBERTa / BERT
        "layers",                  # some DeBERTa variants
        "encoder.layers",          # ModernBERT / RoBERTa
        "layer",
    ):
        obj = encoder
        found = True
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                found = False
                break
        if found and obj is not None and hasattr(obj, "__len__"):
            return obj
    return None


def _get_embeddings(encoder: nn.Module) -> Optional[nn.Module]:
    """Return the embeddings sub-module from an encoder."""
    return getattr(encoder, "embeddings", None)


# ---------------------------------------------------------------------------
# Public freeze functions
# ---------------------------------------------------------------------------

def freeze_embeddings(model: nn.Module) -> int:
    """Freeze all parameters in the encoder's embedding block.

    Returns:
        Number of parameters frozen.
    """
    enc = _get_encoder(model)
    if enc is None:
        print("[freeze] WARNING: could not locate encoder — skip embedding freeze")
        return 0

    emb = _get_embeddings(enc)
    if emb is None:
        print("[freeze] WARNING: encoder has no .embeddings — skip embedding freeze")
        return 0

    n_frozen = 0
    for param in emb.parameters():
        param.requires_grad = False
        n_frozen += param.numel()

    print(f"[freeze] Froze embeddings: {n_frozen:,} parameters")
    return n_frozen


def freeze_encoder_layers(model: nn.Module, n_layers: int) -> int:
    """Freeze the bottom ``n_layers`` transformer blocks of the encoder.

    Args:
        model: The GLiNER2 / CustomExtractor model.
        n_layers: Number of bottom layers to freeze (0 = no-op).

    Returns:
        Number of parameters frozen.
    """
    if n_layers <= 0:
        return 0

    enc = _get_encoder(model)
    if enc is None:
        print("[freeze] WARNING: could not locate encoder — skip layer freeze")
        return 0

    layers = _get_transformer_layers(enc)
    if layers is None:
        print("[freeze] WARNING: could not locate transformer layers — skip layer freeze")
        return 0

    total = len(layers)
    n_to_freeze = min(n_layers, total)
    n_frozen = 0

    for i in range(n_to_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False
            n_frozen += param.numel()

    print(
        f"[freeze] Froze bottom {n_to_freeze}/{total} transformer layers: "
        f"{n_frozen:,} parameters"
    )
    return n_frozen


def print_frozen_summary(model: nn.Module) -> None:
    """Print trainable vs frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen
    pct = 100.0 * trainable / total if total else 0.0
    print(
        f"[freeze] Parameter summary:\n"
        f"  Trainable : {trainable:>12,}  ({pct:.1f}%)\n"
        f"  Frozen    : {frozen:>12,}  ({100-pct:.1f}%)\n"
        f"  Total     : {total:>12,}"
    )
