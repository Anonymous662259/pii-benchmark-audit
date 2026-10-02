import os

from transformers import AutoConfig
from gliner2 import GLiNER2

from models.custom_extractor import CustomExtractor, CustomGLiNER2Config

# model_type values that mean "a checkpoint we trained via CustomExtractor".
_CUSTOM_MODEL_TYPES = {"custom_extractor", "custom_extractor_mmbert"}
# model_type of a bi-encoder checkpoint we trained.
_BI_MODEL_TYPE = "custom_extractor_bi"
# model_type of a two-encoder (two-tower) bi-encoder checkpoint we trained.
_TWO_TOWER_MODEL_TYPE = "custom_extractor_bi_encoder_2_encoder"
# model_type of a native (heads-included) GLiNER2 checkpoint.
_NATIVE_GLINER2_TYPE = "extractor"


def _resolve_path(model_name: str, subfolder: str = None) -> str:
    """If subfolder is given, snapshot_download then return local path/subfolder."""
    if not subfolder:
        return model_name
    from huggingface_hub import snapshot_download
    local_root = snapshot_download(
        repo_id=model_name,
        allow_patterns=[f"{subfolder}/*", f"{subfolder}/**"],
    )
    return os.path.join(local_root, subfolder)


def _model_type_of(path: str) -> str:
    """Return the checkpoint's model_type, read directly from config.json.

    Reads the raw JSON (local dir or HF repo) rather than via AutoConfig, so it
    works for model_types not yet registered with transformers in this process
    (e.g. our ``custom_extractor_bi`` before ``models.bi_encoder`` is imported —
    otherwise AutoConfig.from_pretrained raises and detection silently fails).
    """
    import json
    cfg_path = None
    local = os.path.join(path, "config.json")
    if os.path.isfile(local):
        cfg_path = local
    else:
        try:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(path, "config.json")
        except Exception:
            cfg_path = None
    if cfg_path:
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return json.load(f).get("model_type", "") or ""
        except Exception:
            pass
    try:  # last resort
        return getattr(AutoConfig.from_pretrained(path), "model_type", "") or ""
    except Exception:
        return ""


def build_fresh_on_backbone(encoder_id: str, max_width: int = 30, arch: str = "uni",
                            schema_projection_dim=256):
    """Cold-start a fresh GLiNER2 on a raw encoder (task heads random-init).

    arch="uni" → CustomExtractor (joint text+schema encoding).
    arch="bi"  → BiEncoderExtractor (separate text/label encoding, shared backbone).
                 ``schema_projection_dim`` = shared projection dim D (int, or
                 "hidden" → D = backbone hidden size, matching uni for a fair
                 comparison). Ignored for uni.
    arch="bi-encoder-2-encoder" → TwoTowerExtractor (two separate encoders,
                 siamese init, both trained). Same projection semantics as bi.
    """
    if arch == "bi-encoder-2-encoder":
        from models.two_tower import TwoTowerConfig, TwoTowerExtractor
        config = TwoTowerConfig(model_name=encoder_id, max_width=max_width,
                                schema_projection_dim=schema_projection_dim)
        model = TwoTowerExtractor(config)
    elif arch == "bi":
        from models.bi_encoder import BiEncoderConfig, BiEncoderExtractor
        config = BiEncoderConfig(model_name=encoder_id, max_width=max_width,
                                 schema_projection_dim=schema_projection_dim)
        model = BiEncoderExtractor(config)
    else:
        config = CustomGLiNER2Config(model_name=encoder_id, max_width=max_width)
        model = CustomExtractor(config)

    # Force uniform fp32 across encoder + freshly-built task heads. Some encoders
    # load in fp16 (encoder=Half) while the new heads (span_rep/classifier/…) are
    # fp32 → "mat1 and mat2 must have the same dtype: Half and Float" once AMP
    # autocast is off (e.g. deberta-v2 trains in fp32). Keeping fp32 master
    # weights is the correct AMP setup; the trainer's autocast still casts to
    # bf16/fp16 per-forward for backbones that train in half precision.
    return model.float()


def _identity_init_(linear):
    """In-place: make an nn.Linear an identity map (W=I, b=0). Requires square."""
    import torch
    with torch.no_grad():
        w = linear.weight
        assert w.shape[0] == w.shape[1], (
            f"identity init needs a square Linear, got {tuple(w.shape)} — "
            "warm-start requires proj-dim=hidden (D=H)."
        )
        w.copy_(torch.eye(w.shape[0], dtype=w.dtype, device=w.device))
        if linear.bias is not None:
            linear.bias.zero_()


def build_warmstart_bi2enc_from_gliner2(
    source_checkpoint: str,
    arch: str = "bi-encoder-2-encoder",
    max_width: int = 30,
):
    """Cold-start a bi / two-tower GLiNER2, then WARM-START its weights from a
    native GLiNER2 checkpoint (heads-included, e.g. ``fastino/gliner2-multi-v1``).

    Unlike the plain backbone path (encoder reused, task heads random-init), this
    copies the source's **encoder AND the four task heads** (span_rep, classifier,
    count_pred, count_embed) into the bi/two-tower model, and initialises the
    bi-encoder projections (text_proj/schema_proj) to **identity** so the projected
    reps equal the encoder output at step 0 (warm-start preserved).

    Requirements / conflicts handled:
      * D must equal H → built with ``schema_projection_dim="hidden"`` (square
        projections, head shapes match the source's H-dim heads).
      * counting_layer is taken from the source (fastino → ``count_lstm`` v1) so
        count_embed copies cleanly (bi default would be v2 → shape/key mismatch).
      * two-tower: label_encoder is re-synced from the warm-started encoder.

    Caveat (not a bug): the source heads were trained on JOINT-encoded reps
    (text+labels with cross-attention); bi/two-tower encode them SEPARATELY, so
    this is a strong init, not a no-op equivalent — fine-tuning is still needed.
    """
    from gliner2 import GLiNER2

    src = GLiNER2.from_pretrained(source_checkpoint)
    base_id = src.config.model_name                      # microsoft/mdeberta-v3-base
    counting = getattr(src.config, "counting_layer", "count_lstm")

    if arch == "bi-encoder-2-encoder":
        from models.two_tower import TwoTowerConfig, TwoTowerExtractor
        config = TwoTowerConfig(model_name=base_id, max_width=max_width,
                                counting_layer=counting, schema_projection_dim="hidden")
        model = TwoTowerExtractor(config)
    elif arch == "bi":
        from models.bi_encoder import BiEncoderConfig, BiEncoderExtractor
        config = BiEncoderConfig(model_name=base_id, max_width=max_width,
                                 counting_layer=counting, schema_projection_dim="hidden")
        model = BiEncoderExtractor(config)
    else:
        raise ValueError(f"warm-start only supports bi / bi-encoder-2-encoder, got {arch!r}")

    # ── warm-start: copy encoder + the four task heads (shapes match: D=H) ──
    missing = model.encoder.load_state_dict(src.encoder.state_dict(), strict=False)
    if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
        print(f"[warmstart] encoder copy: missing={list(missing.missing_keys)} "
              f"unexpected={list(missing.unexpected_keys)}")
    for head in ("span_rep", "classifier", "count_pred", "count_embed"):
        getattr(model, head).load_state_dict(getattr(src, head).state_dict())
        print(f"[warmstart] copied head: {head}")

    # ── identity-init projections so warm-started heads see the encoder output ──
    _identity_init_(model.text_proj)
    _identity_init_(model.schema_proj)
    print("[warmstart] text_proj / schema_proj set to identity (D=H)")

    # ── two-tower: re-sync the label tower from the warm-started text encoder ──
    if hasattr(model, "label_encoder"):
        model.label_encoder.load_state_dict(model.encoder.state_dict())
        print("[warmstart] label_encoder re-synced from warm-started encoder")

    del src
    return model.float()


def load_gliner2_model(
    model_name: str,
    max_width: int = 30,
    use_mmbert: bool = False,
    backbone: str = None,
    arch: str = "uni",
    schema_projection_dim=256,
    subfolder: str = None,
    override_max_width: bool = True,
):
    """Load a GLiNER2 model.

    Three paths, auto-selected:
      1. ``backbone`` given (a key in config.backbones) → cold-start a fresh
         uni-encoder on that backbone's encoder (training the comparison set).
      2. ``model_name`` is a checkpoint we trained (model_type custom_extractor*)
         → ``CustomExtractor.from_pretrained`` (round-trip for eval/predict).
      3. ``model_name`` is a native GLiNER2 checkpoint → ``GLiNER2.from_pretrained``;
         a raw HF encoder → cold-start on it.

    Args:
        model_name: HF repo id or local path (ignored if ``backbone`` is set).
        max_width:  Max span width (applied when override_max_width=True).
        use_mmbert: Back-compat flag; forces the CustomExtractor path.
        backbone:   Optional key into config.backbones.BACKBONES.
        subfolder:  Optional subfolder inside the HF repo (e.g. "best").
    """
    # ── Path 1: fresh build on a registry backbone (uni-encode experiment) ──
    if backbone is not None:
        from config.backbones import get_backbone, resolve_encoder
        bb = get_backbone(backbone)
        encoder_id = resolve_encoder(bb)
        model = build_fresh_on_backbone(encoder_id, max_width=max_width, arch=arch,
                                        schema_projection_dim=schema_projection_dim)
        if override_max_width:
            _apply_max_width(model, max_width)
        return model

    load_path = _resolve_path(model_name, subfolder)
    mt = _model_type_of(load_path)

    # ── Path 2c: round-trip a two-encoder (two-tower) checkpoint ────────────
    if mt == _TWO_TOWER_MODEL_TYPE:
        from models.two_tower import TwoTowerExtractor
        model = TwoTowerExtractor.from_pretrained(load_path)
    # ── Path 2b: round-trip a bi-encoder checkpoint ─────────────────────────
    elif mt == _BI_MODEL_TYPE:
        from models.bi_encoder import BiEncoderExtractor
        model = BiEncoderExtractor.from_pretrained(load_path)
    # ── Path 2: round-trip a CustomExtractor checkpoint ─────────────────────
    elif use_mmbert or mt in _CUSTOM_MODEL_TYPES:
        if mt in _CUSTOM_MODEL_TYPES:
            model = CustomExtractor.from_pretrained(load_path)
        else:
            # Fresh build from a raw encoder explicitly requested via use_mmbert.
            model = build_fresh_on_backbone(load_path, max_width=max_width)
    # ── Path 3a: native GLiNER2 checkpoint (heads included) ─────────────────
    elif mt == _NATIVE_GLINER2_TYPE:
        model = GLiNER2.from_pretrained(load_path)
    # ── Path 3b: a bare HF encoder → cold-start a GLiNER2 on it ─────────────
    else:
        model = build_fresh_on_backbone(load_path, max_width=max_width)

    if override_max_width:
        _apply_max_width(model, max_width)
    # Reloaded checkpoints can come back with a fp16 encoder while the heads are
    # fp32 (observed: TwoTowerExtractor.from_pretrained rebuilds the mDeBERTa
    # encoder in float16 on this image). For deberta-v2 the fp16 disentangled
    # attention then degenerates at inference → the model predicts nothing
    # (F1=0) even though training/eval_loss were healthy. Force uniform fp32
    # master weights; evaluate.py re-applies .half()/ONNX afterwards per
    # --backend, so half/quantized backends are unaffected.
    return model.float()


def _apply_max_width(model, max_width: int):
    model.config.max_width = max_width
    model.max_width = max_width
    model.span_rep.span_rep_layer.max_width = max_width
