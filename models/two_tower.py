"""TwoTowerExtractor — two-encoder bi-encoder (``--arch bi-encoder-2-encoder``).

Unlike the shared-weight bi-encoder (``BiEncoderExtractor``: ONE backbone encodes
both text and labels), this variant uses **two separate encoder instances** that
start from the **same pretrained weights** (siamese init) and then **diverge**
during training:

    text  tower → self.encoder        (gradients from span/text)
    label tower → self.label_encoder  (gradients from label descriptions)

Both towers are trained (no freezing) and both land in the optimizer's
``encoder_lr`` group (``trainer.py`` groups by the substring ``"encoder"``, which
matches ``label_encoder.*`` too). Divergence under different gradient signals is
the whole point — without it this would collapse back to the shared-weight bi.

This mirrors the classic GLiNER two-tower bi-encoder (Knowledgator's
``gliner-bi-*``: a dedicated ``labels_encoder`` distinct from the text encoder),
but is **siamese-init / homogeneous**: same backbone, same tokenizer, identical
init. So the processor, projections (``text_proj`` / ``schema_proj``), span/score
path, and label cache are all reused unchanged from ``BiEncoderExtractor`` — the
only override is *which encoder* the label branch runs through.

    text  → encoder        → text_proj  (H→D) → span_rep(D) → S
    label → label_encoder  → schema_proj(H→D) → E
    score: O = einsum('lkd,cd->lkc', S, E)

Because both encoders share the backbone hidden size H, ``schema_proj`` (H→D)
applies to the label tower's output unchanged. D defaults to the backbone hidden
size (no bottleneck), mirroring GLiNER's two-tower which projects each encoder to
a common working dim; override with ``--proj-dim``.
"""
from transformers import AutoConfig

from models.bi_encoder import BiEncoderConfig, BiEncoderExtractor

# model_type stored in config.json (underscores — transformers convention avoids
# '-'); the user-facing --arch token stays "bi-encoder-2-encoder".
TWO_TOWER_MODEL_TYPE = "custom_extractor_bi_encoder_2_encoder"


class TwoTowerConfig(BiEncoderConfig):
    model_type = TWO_TOWER_MODEL_TYPE
    encoder_mode = "bi2enc"
    # Inherits schema_projection_dim/cache_labels/etc. from BiEncoderConfig. The
    # default D (=256) is overridden to "hidden" by the loader/train CLI for this
    # arch, but a concrete int is always persisted by BiEncoderExtractor.__init__.


try:
    AutoConfig.register(TWO_TOWER_MODEL_TYPE, TwoTowerConfig)
except (TypeError, ValueError):
    pass


class TwoTowerExtractor(BiEncoderExtractor):
    """Bi-encoder with two separate (siamese-init) encoders, both trained."""

    config_class = TwoTowerConfig

    def __init__(self, config, tokenizer=None, encoder_config=None, **kwargs):
        # Builds self.encoder + projections + heads (shared-weight bi machinery).
        super().__init__(config, tokenizer=tokenizer, encoder_config=encoder_config, **kwargs)

        # ── second encoder: SAME backbone, SAME init as self.encoder ──
        # Built via the identical _load_encoder call (round-trip uses from_config →
        # no re-download; cold-start uses from_pretrained). After resizing to the
        # same vocab, we COPY self.encoder's full state into it: resize_token_
        # embeddings random-inits the *new* special-token rows independently per
        # call, so without this copy the two encoders would differ on those rows.
        # The copy makes them bit-identical at init while staying separate trainable
        # tensors that diverge during training. Harmless on round-trip — the outer
        # from_pretrained's load_state_dict(strict) overwrites both from the
        # checkpoint afterwards.
        self.label_encoder = self._load_encoder(config.model_name, encoder_config)
        self.label_encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.label_encoder.load_state_dict(self.encoder.state_dict())

    def _encode_label_ids(self, ids_list, device):
        """Route the label branch through the SEPARATE, independently-trained
        label encoder (shared-weight bi reuses self.encoder here)."""
        return self._encode_ids(ids_list, device, encoder=self.label_encoder)
