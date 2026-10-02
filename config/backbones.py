"""Backbone registry for the uni-encoder backbone-comparison experiment.

Each entry describes ONE encoder we want to train a fresh GLiNER2 (uni-encoder)
on top of. Following the experiment design (see Conflict A), every backbone is
treated as a *raw encoder*: we keep only its encoder weights and cold-start all
GLiNER2 task heads, so the comparison isolates the encoder.

Two kinds of source:
  * "default weights"            → a vanilla HF encoder (raw baseline).
  * pretrained-on-NER/PII variant → we EXTRACT the fine-tuned encoder from a
                                     GLiNER/token-classifier checkpoint so the
                                     entry differs meaningfully from its base.

NOTE: two backbone labels were corrected here vs the original scratch list —
  * "mmBERT (joneauxedgar/pasteproof-...)"  → it is actually ModernBERT-base.
  * "ModernBERT (hivetrace/gliner-guard)"   → its backbone is mmBERT-small.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# load_kind:
#   "raw"                 → encoder_id is directly AutoModel-loadable.
#   "automodel_drop_head" → encoder_id is a *ForTokenClassification etc. repo;
#                           AutoModel.from_pretrained loads the base encoder and
#                           silently drops the task head (kept PII pretraining).
#   "extract_gliner2"     → source is a GLiNER2 checkpoint; load via gliner2 and
#                           save `.encoder` (+ tokenizer) to a local cache dir.
#   "extract_gliner_v1"   → source is a GLiNER v1 checkpoint (package `gliner`).
@dataclass(frozen=True)
class Backbone:
    key: str
    arch: str                 # "deberta-v2" | "modernbert"  (informational)
    encoder_id: str           # AutoModel-loadable id, OR base used by the extractor
    hidden_size: int          # informational; heads auto-derive from encoder
    encoder_max_tokens: int   # encoder's subword position ceiling
    load_kind: str = "raw"
    source_checkpoint: Optional[str] = None  # set for pretrained-on-NER/PII variants
    # ── per-backbone TRAINING overrides (applied on top of DEFAULT_TRAINING) ──
    max_len: int = 2048       # word-level (WhitespaceTokenSplitter units)
    dtype: str = "bf16"       # C.2: uni-encode experiment defaults to bf16
    num_epochs: int = 6       # D.10: cold-start heads → default 6 epochs
    multilingual: bool = True
    notes: str = ""

    @property
    def chunk_tokens(self) -> int:
        """Eval chunk size mirrors training max_len (same per-variant budget)."""
        return self.max_len


# DeBERTa family: max_len kept at 1024 to match the original GLiNER2 training
# budget and keep all variants consistent (eval chunk_tokens mirrors this).
# DeBERTa-v3 has encoder_max_tokens=512 but position_biased_input=false, so it
# does NOT crash beyond 512 (relative+bucketed positions) — quality just softens
# past the pretrained range. See Conflict C.1.
_DEBERTA_MAXLEN = 1024

BACKBONES: Dict[str, Backbone] = {
    # ── DeBERTa / mDeBERTa (deberta-v2 arch, 512-pos) ────────────────────────
    # NOTE: DeBERTa-v2/v3 disentangled attention is numerically unstable in half
    # precision — fp16 overflows, and bf16 (7-bit mantissa) still NaNs under the
    # large cold-start-head gradients here (observed: loss=NaN by step ~171 even
    # at lr≈0). So the whole deberta-v2 family defaults to dtype="fp32". Override
    # with --dtype to experiment. modernbert/mmbert keep the bf16 default.
    "deberta_base": Backbone(
        key="deberta_base", arch="deberta-v2",
        encoder_id="microsoft/deberta-v3-base",
        hidden_size=768, encoder_max_tokens=512,
        load_kind="raw", max_len=_DEBERTA_MAXLEN, multilingual=False,
        dtype="fp32",
        notes="English-centric DeBERTa-v3; weak on de/vi. Vanilla baseline.",
    ),
    "mdeberta_base": Backbone(
        key="mdeberta_base", arch="deberta-v2",
        encoder_id="microsoft/mdeberta-v3-base",
        hidden_size=768, encoder_max_tokens=512,
        load_kind="raw", max_len=_DEBERTA_MAXLEN,
        dtype="fp32",
        notes="Multilingual DeBERTa-v3. Vanilla baseline; also the base of "
              "fastino/gliner2-multi-v1 and urchade/gliner_multi_pii-v1.",
    ),
    "mdeberta_gliner2_fastino": Backbone(
        key="mdeberta_gliner2_fastino", arch="deberta-v2",
        encoder_id="microsoft/mdeberta-v3-base",   # base used for extraction fallback
        hidden_size=768, encoder_max_tokens=512,
        load_kind="extract_gliner2",
        source_checkpoint="fastino/gliner2-multi-v1",
        max_len=_DEBERTA_MAXLEN,
        dtype="fp32",
        notes="Encoder extracted from the GLiNER2 multi PII checkpoint "
              "(mDeBERTa-v3-base backbone, GLiNER2-adapted).",
    ),
    "deberta_gliner2_base_fastino": Backbone(
        key="deberta_gliner2_base_fastino", arch="deberta-v2",
        encoder_id="microsoft/deberta-v3-base",   # base used for extraction fallback
        hidden_size=768, encoder_max_tokens=512,
        load_kind="extract_gliner2",
        source_checkpoint="fastino/gliner2-base-v1",
        max_len=_DEBERTA_MAXLEN,
        dtype="fp32", multilingual=False,
        notes="Encoder extracted from the GLiNER2 base-v1 checkpoint "
              "(microsoft/deberta-v3-base backbone, English-centric, GLiNER2-adapted). "
              "GLiNER2-adapted counterpart of the raw 'deberta_base' entry.",
    ),
    "mdeberta_gliner_pii_urchade": Backbone(
        key="mdeberta_gliner_pii_urchade", arch="deberta-v2",
        encoder_id="microsoft/mdeberta-v3-base",
        hidden_size=768, encoder_max_tokens=512,
        load_kind="extract_gliner_v1",
        source_checkpoint="urchade/gliner_multi_pii-v1",
        max_len=_DEBERTA_MAXLEN,
        dtype="fp32",
        notes="GLiNER **v1** (package `gliner`, not gliner2). Encoder is "
              "mDeBERTa-v3-base fine-tuned on synthetic PII NER.",
    ),

    # ── ModernBERT / mmBERT (modernbert arch, 8192-pos, RoPE) ────────────────
    "modernbert_base": Backbone(
        key="modernbert_base", arch="modernbert",
        encoder_id="answerdotai/ModernBERT-base",
        hidden_size=768, encoder_max_tokens=8192,
        load_kind="raw", max_len=2048, multilingual=False,
        notes="English-only ModernBERT. Vanilla baseline; weak on de/vi.",
    ),
    "modernbert_pasteproof": Backbone(
        key="modernbert_pasteproof", arch="modernbert",
        encoder_id="joneauxedgar/pasteproof-pii-detector-v2",
        hidden_size=768, encoder_max_tokens=8192,
        load_kind="automodel_drop_head",
        source_checkpoint="joneauxedgar/pasteproof-pii-detector-v2",
        max_len=2048, multilingual=False,
        notes="ModernBERT-base continued-pretrained on PII (BIO token-cls). "
              "AutoModel drops the BIO head and keeps the PII-adapted encoder.",
    ),
    "mmbert_base": Backbone(
        key="mmbert_base", arch="modernbert",
        encoder_id="jhu-clsp/mmBERT-base",
        hidden_size=768, encoder_max_tokens=8192,
        load_kind="raw", max_len=2048,
        notes="Multilingual ModernBERT (mmBERT-base, hidden 768). Vanilla baseline.",
    ),
    "mmbert_small_gliner_guard_hivetrace": Backbone(
        key="mmbert_small_gliner_guard_hivetrace", arch="modernbert",
        encoder_id="jhu-clsp/mmBERT-small",        # base for extraction fallback
        hidden_size=384, encoder_max_tokens=8192,
        load_kind="extract_gliner2",
        source_checkpoint="hivetrace/gliner-guard-uniencoder",
        max_len=2048,
        notes="GLiNER Guard uni-encoder; backbone is mmBERT-SMALL (hidden 384), "
              "fine-tuned on safety+PII. (Old label 'ModernBERT' was wrong.)",
    ),
}


def get_backbone(key: str) -> Backbone:
    if key not in BACKBONES:
        raise KeyError(
            f"Unknown backbone '{key}'. Available: {', '.join(BACKBONES)}"
        )
    return BACKBONES[key]


def training_overrides(bb: Backbone) -> dict:
    """Per-backbone overrides merged on top of DEFAULT_TRAINING in train.py."""
    ov = dict(
        max_len=bb.max_len,
        num_epochs=bb.num_epochs,
        # cold-start heads: lower task_lr + bf16, mirroring the mmBERT recipe
        task_lr=2e-4,
        adam_epsilon=1e-6,
        eval_strategy="epoch",
        early_stopping=True,
        early_stopping_patience=2,
    )
    if bb.dtype == "bf16":
        ov.update(fp16=False, bf16=True)
    elif bb.dtype == "fp16":
        ov.update(fp16=True, bf16=False)
    else:  # "fp32"
        ov.update(fp16=False, bf16=False)
    return ov


def _extract_encoder(bb: Backbone):
    """Return the bare HF encoder (AutoModel) pulled from the source checkpoint."""
    if bb.load_kind == "extract_gliner2":
        from gliner2 import GLiNER2
        return GLiNER2.from_pretrained(bb.source_checkpoint).encoder
    if bb.load_kind == "extract_gliner_v1":
        try:
            from gliner import GLiNER  # original v1 package
        except ImportError as e:
            raise RuntimeError(
                f"{bb.key}: needs the `gliner` (v1) package to extract the "
                f"encoder from {bb.source_checkpoint}. Either `pip install gliner` "
                f"or drop this backbone (its base '{bb.encoder_id}' is already a "
                f"separate raw entry)."
            ) from e
        # verified attribute chain: GLiNER.model.token_rep_layer.bert_layer.model
        return GLiNER.from_pretrained(bb.source_checkpoint).model.token_rep_layer.bert_layer.model
    raise ValueError(f"Unknown load_kind: {bb.load_kind}")


def _mean_abs_delta_vs_base(encoder, base_id: str) -> float:
    """Mean |Δ| of overlapping same-shape params between encoder and its base.

    Guards against a silent failure where the 'extracted' encoder is really the
    untouched base (→ the variant would be identical to the raw entry).
    """
    from transformers import AutoModel
    base = AutoModel.from_pretrained(base_id, trust_remote_code=True)
    ext_sd, base_sd = encoder.state_dict(), base.state_dict()
    diffs = [
        (ext_sd[k] - base_sd[k]).abs().mean().item()
        for k in ext_sd
        if k in base_sd and ext_sd[k].shape == base_sd[k].shape
    ]
    return sum(diffs) / len(diffs) if diffs else 0.0


def resolve_encoder(
    bb: Backbone, cache_dir: str = "./.encoder_cache", verify: bool = True
) -> str:
    """Return an AutoModel-loadable encoder path/id for this backbone.

    For raw / automodel_drop_head this is just ``encoder_id``. For the
    extract_* kinds, the fine-tuned encoder is pulled out of the source
    checkpoint once and cached locally (with the BASE tokenizer so GLiNER2 can
    cleanly re-add its own special tokens), then that local path is returned.
    """
    if bb.load_kind in ("raw", "automodel_drop_head"):
        return bb.encoder_id

    out_dir = os.path.join(cache_dir, bb.key)
    if os.path.isfile(os.path.join(out_dir, "config.json")):
        return out_dir  # already extracted

    encoder = _extract_encoder(bb)

    if verify:
        delta = _mean_abs_delta_vs_base(encoder, bb.encoder_id)
        if delta < 1e-6:
            raise RuntimeError(
                f"{bb.key}: extracted encoder is identical to base "
                f"'{bb.encoder_id}' (mean|Δ|={delta:.2e}). Extraction likely "
                f"failed or hit a base fallback — this variant would duplicate "
                f"the raw entry. Aborting."
            )
        print(f"[extract] {bb.key}: encoder vs base mean|Δ|={delta:.4e} (OK, adapted)")

    os.makedirs(out_dir, exist_ok=True)
    encoder.save_pretrained(out_dir)
    # Save the BASE tokenizer; GLiNER2 re-adds its 10 special tokens idempotently.
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(bb.encoder_id).save_pretrained(out_dir)
    return out_dir
