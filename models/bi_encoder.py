"""BiEncoderExtractor — shared-weight bi-encoder GLiNER2 (`--arch bi`).

Mirrors ``hivetrace/gliner-guard-biencoder``. Unlike the uni-encoder (text + schema
concatenated into one sequence with full cross-attention), the bi-encoder encodes
**text and labels separately through the SAME backbone**, projects both into a shared
``schema_projection_dim`` space, and scores spans against labels by dot-product. Label
embeddings are text-independent → cacheable.

Module names/shapes are kept identical to the hivetrace checkpoint so its weights can be
loaded with ``strict=True`` to validate the architecture (milestone M0). See
``docs/BI_ENCODER_DESIGN.md`` §2 for the exact tensor list this mirrors.

Backbone(H) ──► text_proj  (H→D) ──► span_rep(D) ──► S
            └─► schema_proj(H→D) ──► E ;  O = einsum('BLKD,BCD->BLKC', S, E)
"""
import torch
import torch.nn as nn

from gliner.modeling.span_rep import SpanRepLayer
from gliner2.layers import CountLSTM, CountLSTMoE, CountLSTMv2, create_mlp
from transformers import AutoConfig

from models.custom_extractor import CustomExtractor, CustomGLiNER2Config


class BiEncoderConfig(CustomGLiNER2Config):
    model_type = "custom_extractor_bi"
    # Class-level fallbacks so the fields survive PretrainedConfig round-trips even
    # on transformers versions that drop unknown instance attrs (same defensive
    # pattern as CustomGLiNER2Config.token_pooling / counting_layer — see C3).
    token_pooling = "first"
    counting_layer = "count_lstm_v2"
    encoder_mode = "bi"
    schema_projection_dim = 256
    cross_fuser_layers = 1
    cross_fuser_heads = 0          # 0 → cross-fuser disabled (pure bi-encoder)

    cache_labels = True

    def __init__(self, model_name="bert-base-uncased", max_width=30,
                 counting_layer="count_lstm_v2", token_pooling="first",
                 max_len=None, encoder_mode="bi", schema_projection_dim=256,
                 cross_fuser_layers=1, cross_fuser_heads=0, cache_labels=True, **kwargs):
        super().__init__(
            model_name=model_name, max_width=max_width,
            counting_layer=counting_layer, token_pooling=token_pooling,
            max_len=max_len, **kwargs,
        )
        self.encoder_mode = encoder_mode
        self.schema_projection_dim = schema_projection_dim
        self.cross_fuser_layers = cross_fuser_layers
        self.cross_fuser_heads = cross_fuser_heads
        self.cache_labels = cache_labels


try:
    AutoConfig.register("custom_extractor_bi", BiEncoderConfig)
except (TypeError, ValueError):
    pass


class BiClassifier(nn.Module):
    """Text×label scorer for *classification* tasks (NOT used on entities-only PII).

    Shapes mirror the checkpoint: ``proj_text``/``proj_label`` = Linear(D→2D),
    ``out_mlp`` = MLP(2D+D → 2D → 1). The exact way text and label reps are combined
    before ``out_mlp`` is NOT yet confirmed (design doc §8 C5) — the forward below is
    PROVISIONAL. It does not affect strict state_dict loading (shapes only) and is not
    exercised by the PII (entities) path.
    """

    def __init__(self, d_model: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_text = nn.Linear(d_model, proj_dim)
        self.proj_label = nn.Linear(d_model, proj_dim)
        # input 2D+D matches checkpoint out_mlp.0=[2D, 2D+D]; dropout>0 → indices .0/.3
        self.out_mlp = create_mlp(
            input_dim=proj_dim + d_model, intermediate_dims=[proj_dim],
            output_dim=1, dropout=dropout, activation="relu", add_layer_norm=False,
        )

    def forward(self, text_emb: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        # PROVISIONAL combine: concat[proj_text(text), label_emb]. Revisit in M1.
        u = self.proj_text(text_emb)
        return self.out_mlp(torch.cat([u, label_emb], dim=-1)).squeeze(-1)


class BiEncoderExtractor(CustomExtractor):
    """GLiNER2 bi-encoder: shared backbone, separate text/label projection to D."""

    config_class = BiEncoderConfig

    def __init__(self, config, tokenizer=None, **kwargs):
        # CustomExtractor builds uni heads at hidden_size H; we rebuild them at the
        # shared projection dim D and add the bi-encoder-specific projections.
        super().__init__(config, tokenizer=tokenizer, **kwargs)

        H = self.encoder.config.hidden_size
        # schema_projection_dim may be an int, or "hidden"/"H" → D = backbone H
        # (matches the uni-encoder, which scores in full hidden size — for a fair
        # joint-vs-separate-encoding comparison). Persist the resolved int so
        # save/load round-trips to a concrete dim.
        _pd = getattr(config, "schema_projection_dim", 256)
        D = H if str(_pd).lower() in ("hidden", "h", "none") else int(_pd)
        self.proj_dim = D
        config.schema_projection_dim = D

        # ── bi-encoder projections (H → D) — the core of the architecture ──
        self.text_proj = nn.Linear(H, D)
        self.schema_proj = nn.Linear(H, D)

        # ── task heads, now operating in the shared D-space (not H) ──
        self.span_rep = SpanRepLayer(
            span_mode="markerV0", hidden_size=D,
            max_width=config.max_width, dropout=0.1,
        )
        self.classifier = create_mlp(
            input_dim=D, intermediate_dims=[D * 2], output_dim=1,
            dropout=0.0, activation="relu", add_layer_norm=False,
        )
        self.count_pred = create_mlp(
            input_dim=D, intermediate_dims=[D * 2], output_dim=20,
            dropout=0.0, activation="relu", add_layer_norm=False,
        )
        # count_embed honours config.counting_layer (was hard-coded to v2). Warm-
        # starting heads from a native GLiNER2 checkpoint needs the SAME module the
        # source trained (fastino/gliner2-multi-v1 uses count_lstm v1) so the
        # state_dict copies cleanly. Defaults to v2 to keep cold-start behaviour.
        _cl = getattr(config, "counting_layer", "count_lstm_v2")
        if _cl == "count_lstm":
            self.count_embed = CountLSTM(D)
        elif _cl == "count_lstm_moe":
            self.count_embed = CountLSTMoE(hidden_size=D, n_experts=4, ffn_mult=2, dropout=0.1)
        else:
            self.count_embed = CountLSTMv2(D)
        self.bi_classifier = BiClassifier(D, proj_dim=D * 2, dropout=0.1)

        # Span scorer must live in the shared projection dim D (base __init__ built
        # it at H). dot is unchanged; biaffine/cosine score in D for affine_adapt.
        self._build_span_scorer(D)

        # M2.1 label cache: projected label vectors keyed by schema tokens, filled
        # lazily at inference (text-independent → reusable across chunks/requests).
        self.cache_labels = bool(getattr(config, "cache_labels", True))
        self._label_cache = {}

    def clear_label_cache(self):
        """Drop cached label embeddings (call if the schema or weights change)."""
        self._label_cache = {}

    # =====================================================================
    # Bi-encoder forward: text and labels encoded in SEPARATE passes, each
    # projected H→D. The rest of GLiNER2's forward/loss is dimension-agnostic
    # (it already separates token vs schema embeddings), so overriding ONLY
    # _encode_batch is enough — no processor / loss / PreprocessedBatch edits.
    # =====================================================================

    def _encode_ids(self, ids_list, device, encoder=None):
        """Pad a list of token-id lists, run an encoder (default the text/shared
        encoder), return (B, L, H). ``encoder`` lets the two-encoder variant route
        the label branch through a separate label encoder."""
        enc = encoder if encoder is not None else self.encoder
        maxlen = max((len(x) for x in ids_list), default=0)
        b = len(ids_list)
        input_ids = torch.zeros(b, maxlen, dtype=torch.long, device=device)
        mask = torch.zeros(b, maxlen, dtype=torch.long, device=device)
        for i, ids in enumerate(ids_list):
            if ids:
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                mask[i, : len(ids)] = 1
        out = enc(input_ids=input_ids, attention_mask=mask)
        h = getattr(out, "last_hidden_state", None)
        return h if h is not None else out[0]

    def _encode_label_ids(self, ids_list, device):
        """Encode label-only sequences. Shared-weight bi uses the SAME backbone as
        text (self.encoder); the two-encoder variant (TwoTowerExtractor) overrides
        this to route through a separate, independently-trained label encoder."""
        return self._encode_ids(ids_list, device)

    def _encode_batch(self, batch):
        """Return (all_token_embs[D], all_schema_embs[D]) — same structure as the
        uni path's processor.extract_embeddings_from_batch, but computed via two
        text-independent encoder passes and projected into the shared D-space."""
        device = next(self.parameters()).device
        proc = self.processor
        n = len(batch)

        # ── TEXT branch: one pass over text-only sequences (no schema tokens) ──
        text_ids, text_pos = [], []
        for i in range(n):
            fmt = proc._format_input_with_mapping([], batch.text_tokens[i])
            text_ids.append(fmt["input_ids"])
            text_pos.append(fmt["text_word_first_positions"])
        th = self._encode_ids(text_ids, device)              # (n, Lt, H)
        all_token_embs = []
        for i in range(n):
            if text_pos[i]:
                idx = torch.tensor(text_pos[i], dtype=torch.long, device=device)
                words = th[i].index_select(0, idx)           # (n_words, H)
            else:
                words = th.new_zeros(0, th.shape[-1])
            all_token_embs.append(self.text_proj(words))     # (n_words, D)

        # ── LABEL branch: encode each schema separately (text-independent). At
        #    inference, projected label vectors are cached by schema tokens so a
        #    fixed schema (e.g. the 44 PII labels) is encoded ONCE and reused across
        #    all chunks — the bi-encoder's throughput advantage. Disabled while
        #    training (labels must keep gradients).
        use_cache = self.cache_labels and not self.training
        all_schema_embs = [[[] for _ in range(batch.schema_counts[i])] for i in range(n)]
        sch_ids, sch_special, sch_ij = [], [], []
        for i in range(n):
            for j, sch_tokens in enumerate(batch.schema_tokens_list[i]):
                if use_cache:
                    hit = self._label_cache.get(tuple(sch_tokens))
                    if hit is not None:
                        all_schema_embs[i][j] = hit
                        continue
                fmt = proc._format_input_with_mapping([sch_tokens], [])
                sch_ids.append(fmt["input_ids"])
                sch_special.append(fmt["schema_special_positions"][0])
                sch_ij.append((i, j))
        if sch_ids:
            sh = self._encode_label_ids(sch_ids, device)     # (R, Ls, H)
            for row, (i, j) in enumerate(sch_ij):
                positions = sch_special[row]
                if not positions:
                    continue
                idx = torch.tensor(positions, dtype=torch.long, device=device)
                vecs = self.schema_proj(sh[row].index_select(0, idx))  # (n_special, D)
                lst = [vecs[k] for k in range(vecs.shape[0])]
                all_schema_embs[i][j] = lst
                if use_cache:
                    self._label_cache[tuple(batch.schema_tokens_list[i][j])] = [v.detach() for v in lst]
        return all_token_embs, all_schema_embs

    def _extract_from_batch(self, batch, threshold, metadata_list,
                            include_confidence, include_spans):
        """Inference counterpart of forward: the engine's _extract_from_batch
        inlines the *uni* joint encoding (text+schema in one pass, no projection),
        which is wrong for bi and would crash (span_rep expects D, gets H). We copy
        its structure but source embeddings from ``self._encode_batch`` (separate
        text/label passes, projected H→D). Everything downstream is dim-agnostic."""
        all_token_embs, all_schema_embs = self._encode_batch(batch)

        span_samples = []
        for i in range(len(batch)):
            has_span = any(t != "classifications" for t in batch.task_types[i])
            if has_span and all_token_embs[i].numel() > 0:
                span_samples.append(i)

        all_span_info = [None] * len(batch)
        if span_samples:
            span_results = self.compute_span_rep_batched([all_token_embs[i] for i in span_samples])
            for idx, si in zip(span_samples, span_results):
                all_span_info[idx] = si

        results = []
        for i in range(len(batch)):
            try:
                results.append(self._extract_sample(
                    token_embs=all_token_embs[i], schema_embs=all_schema_embs[i],
                    schema_tokens_list=batch.schema_tokens_list[i],
                    task_types=batch.task_types[i], text_tokens=batch.text_tokens[i],
                    original_text=batch.original_texts[i], schema=batch.original_schemas[i],
                    start_mapping=batch.start_mappings[i], end_mapping=batch.end_mappings[i],
                    threshold=threshold, metadata=metadata_list[i],
                    include_confidence=include_confidence, include_spans=include_spans,
                    span_info=all_span_info[i],
                ))
            except Exception as e:
                print(f"Error extracting sample {i}: {e}")
                results.append({})
        return results

    # ── lightweight helpers used by M0 validation ──

    def encode_text_spans(self, hidden: torch.Tensor) -> dict:
        """hidden: (text_len, H) word/token embeddings → span_rep in D.

        Returns the dict from Extractor.compute_span_rep (span_rep is (text_len,
        max_width, D)), after projecting tokens H→D.
        """
        return self.compute_span_rep(self.text_proj(hidden))

    def project_labels(self, pooled_label_hidden: torch.Tensor) -> torch.Tensor:
        """pooled_label_hidden: (num_labels, H) → (num_labels, D)."""
        return self.schema_proj(pooled_label_hidden)

    @staticmethod
    def score_spans(span_rep: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        """span_rep (L, K, D) · label_emb (C, D) → (L, K, C) span×label logits."""
        return torch.einsum("lkd,cd->lkc", span_rep, label_emb)
