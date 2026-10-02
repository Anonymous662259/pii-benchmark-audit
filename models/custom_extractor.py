"""CustomExtractor: build a fresh GLiNER2 (uni-encoder) on an arbitrary backbone.

Originally written for mmBERT-base only; now generalized so any HF encoder can
be used as the backbone (DeBERTa, mDeBERTa, ModernBERT, mmBERT, …). The task
heads (span_rep, classifier, count_pred, count_embed) are re-initialized from
the encoder's actual ``hidden_size`` so they match whatever backbone is loaded.

Checkpoints trained this way carry ``model_type = "custom_extractor"``. The old
``"custom_extractor_mmbert"`` model_type is still registered as an alias so
previously trained mmBERT checkpoints keep loading.
"""
from gliner2 import GLiNER2
from gliner2.layers import CountLSTM, create_mlp
from gliner2.model import ExtractorConfig
from gliner2.processor import PreprocessedBatch
from gliner.modeling.span_rep import SpanRepLayer
from transformers import AutoConfig


class CustomGLiNER2Config(ExtractorConfig):
    model_type = "custom_extractor"
    # Class-level fallbacks: some transformers versions (seen on the Modal image)
    # don't retain these as instance attrs through PretrainedConfig.__init__, and
    # Extractor.__init__ reads config.token_pooling / counting_layer. Class attrs
    # are returned by __getattribute__ when the instance dict lacks them.
    token_pooling = "first"
    counting_layer = "count_lstm"

    def __init__(self, model_name="bert-base-uncased", max_width=8,
                 counting_layer="count_lstm", token_pooling="first",
                 max_len=None, **kwargs):
        super().__init__(
            model_name=model_name, max_width=max_width,
            counting_layer=counting_layer, token_pooling=token_pooling,
            max_len=max_len, **kwargs,
        )
        # Re-assert after super() so they are always present as instance attrs.
        self.token_pooling = token_pooling
        self.counting_layer = counting_layer


for _mt in ("custom_extractor", "custom_extractor_mmbert"):
    try:
        AutoConfig.register(_mt, CustomGLiNER2Config)
    except (TypeError, ValueError):
        pass


class CustomExtractor(GLiNER2):
    """GLiNER2 with a swappable backbone and dynamically sized task heads."""

    @classmethod
    def _can_set_experts_implementation(cls):
        return False

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(config, tokenizer=tokenizer, **kwargs)
        # super().__init__ already: loads encoder via AutoModel.from_pretrained,
        # creates processor (SchemaTransformer), resizes token embeddings.
        hidden_size = self.encoder.config.hidden_size  # auto: 768, 384, 1024, …

        self.span_rep = SpanRepLayer(
            span_mode="markerV0",
            hidden_size=hidden_size,
            max_width=config.max_width,
            dropout=0.1,
        )
        self.classifier = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=1,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )
        self.count_pred = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=20,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )
        self.count_embed = CountLSTM(hidden_size)

    def _encode_batch(self, batch: PreprocessedBatch):
        # Pass only input_ids + attention_mask (no token_type_ids) — works for
        # DeBERTa, ModernBERT and mmBERT alike. Mirrors GLiNER2's own default.
        outputs = self.encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:  # some encoders return a plain tuple
            hidden = outputs[0]
        return self.processor.extract_embeddings_from_batch(
            hidden, batch.input_ids, batch
        )
