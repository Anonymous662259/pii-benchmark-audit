import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanScorer(nn.Module):
    """Configurable (span x label) scorer used by the span-extraction heads.

    Replaces the hard-coded ``einsum('lkd,bpd->bplk', span_rep, struct_proj)``
    so the span<->label matching function can be swapped via config for the
    ``feat/affine_adapt`` experiments. ``score_mode="dot"`` is the default and
    is **bit-identical** to the original einsum (no params, no behaviour change),
    so existing checkpoints keep loading with ``strict=True``.

    Shapes (B = #instances/count, P = #labels, L = #start positions, K = widths):
        span_rep    : (L, K, D)
        struct_proj : (B, P, D)
        returns     : (B, P, L, K)  raw logits (sigmoid applied by the caller)

    Modes:
      - ``dot``      : span . label                              (W = I, baseline)
      - ``cosine``   : cosine(span, label) * temperature         (unit-norm + scale)
      - ``biaffine`` : span^T W label  [+ affine offsets]        (Yu 2020)
            * ``biaffine_rank=0``  -> full ``W`` (D x D), optional identity init
            * ``biaffine_rank=r>0``-> low-rank ``W = A B^T`` (A,B = D x r)
            * ``biaffine_residual``-> add the plain dot (low-rank learns a residual)
            * ``biaffine_affine``  -> add ``u_span . span + u_label . label + b``
                                       (this is what makes it *affine*, not just
                                        bilinear)
      - ``lams``     : dot + label-modulated low-rank residual (TuckER/hypernetwork
                       style — see references/survey_affine_scorer_methods.md §2 idea A):
                         u = span @ U ; v = label @ V ; gate = tanh(label @ W_g)
                         logits = dot(span, label) + sum_r u * (gate * v)
                       ``U``/``V`` are a **shared core** across every label (unlike full
                       biaffine's per-model W, which is still shared but dense/D²);
                       ``gate`` is *per-label*, ``W_g`` init 0 so ``gate=0`` at init
                       -> bit-identical to dot (safe warm-start), and only the labels
                       that need extra capacity learn a non-zero gate.
            * ``lams_rank=r``  -> D x r core (default 32); cost ~3*D*r params.

    ``boundary_bias`` (composable with ANY mode above): additive bias over the
    (start, width) plane — ``bias_start[start] + bias_width[width]`` — to give the
    scorer direct boundary-position signal without touching span_rep pooling
    (``SpanRepLayer`` is a third-party ``gliner`` module, not forked here).
    ``bias_width`` is a small embedding table (bounded by ``max_width``);
    ``bias_start`` is a learned linear read-out over a fixed sinusoidal encoding of
    the (unbounded) start index, so it needs no fixed-length table. Both init to
    0 -> no-op until trained.

    ``boundary_token`` (composable with ANY mode above; lattice-competition design):
    label-conditioned read-out of the span's *actual boundary tokens* — unlike
    ``boundary_bias`` (position-only, label-agnostic), this looks at the content of
    the first/last word of the span:

        score += u_start . (h[start] * label)  +  u_end . (h[start + width] * label)

    so the scorer can learn e.g. "TIME spans end in 'afternoon/morning', DATE spans
    don't" — direct signal on the boundary-confusion failure mode. Requires the
    caller to pass ``token_embs`` (word-level embeddings, same first dim as
    ``span_rep``) to :meth:`forward`; the model threads them via ``span_info``.
    ``u_start``/``u_end`` init 0 -> no-op until trained (safe warm-start).
    Token embeddings live in the encoder hidden space, so this requires the scorer
    dim to equal the hidden size (uni arch / ``proj_dim=hidden``).

    ``null_score`` (composable with ANY mode above; set-consistent design): a
    learnable, label-conditioned *null anchor* subtracted from every logit:

        logits -= label . u_null + null_bias

    Decoding at sigmoid > 0.5 then means "the span beats its label's null" — the
    absolute decode threshold disappears from the system (the fix for the
    measured score-compression artifact: hinge/competition losses compress
    logits and a fixed 0.7 threshold silently cuts recall). ``u_null`` init 0
    and ``null_bias`` init ``logit(null_init_p)`` (default 0.7) make a
    warm-started model decode EXACTLY like the legacy fixed-0.7 rule at step 0;
    training then learns the per-label operating point end-to-end.
    ⚠️ Checkpoints trained with ``null_score`` must be evaluated with
    ``--threshold 0.5`` (the null absorbs the operating point).

    Cache-safe for the bi-encoder: ``W``/``B``/lams core/``u_null`` are applied
    to the (text-independent) label side, so projected label vectors stay
    cacheable (``boundary_token`` adds a text-side term but keeps label vectors
    intact).
    """

    def __init__(
        self,
        dim: int,
        score_mode: str = "dot",
        biaffine_rank: int = 0,
        biaffine_residual: bool = True,
        biaffine_affine: bool = False,
        biaffine_init_identity: bool = True,
        learn_temperature: bool = False,
        temp_init: float = 0.0,
        lams_rank: int = 32,
        boundary_bias: bool = False,
        max_width: int = 30,
        boundary_pos_dim: int = 16,
        boundary_token: bool = False,
        null_score: bool = False,
        null_init_p: float = 0.7,
    ):
        super().__init__()
        self.dim = int(dim)
        self.score_mode = score_mode
        self.biaffine_rank = int(biaffine_rank or 0)
        self.biaffine_residual = bool(biaffine_residual)
        self.biaffine_affine = bool(biaffine_affine)
        self.lams_rank = int(lams_rank or 32)
        self.boundary_bias = bool(boundary_bias)
        self.max_width = int(max_width or 30)
        self.boundary_pos_dim = int(boundary_pos_dim or 16)
        self.boundary_token = bool(boundary_token)
        self.null_score = bool(null_score)
        self.null_init_p = float(null_init_p or 0.7)

        if score_mode == "biaffine":
            if self.biaffine_rank > 0:
                self.A = nn.Parameter(torch.empty(self.dim, self.biaffine_rank))
                self.B = nn.Parameter(torch.empty(self.dim, self.biaffine_rank))
                nn.init.xavier_uniform_(self.A)
                nn.init.xavier_uniform_(self.B)
            else:
                if biaffine_init_identity:
                    W = torch.eye(self.dim)            # warm-start == dot product
                else:
                    W = torch.empty(self.dim, self.dim)
                    nn.init.xavier_uniform_(W)
                self.W = nn.Parameter(W)

        elif score_mode == "lams":
            r = self.lams_rank
            self.U = nn.Parameter(torch.empty(self.dim, r))
            self.V = nn.Parameter(torch.empty(self.dim, r))
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
            # Gate init 0 -> tanh(0)=0 -> the whole residual is a no-op at init,
            # regardless of U/V, so lams starts bit-identical to dot.
            self.W_g = nn.Parameter(torch.zeros(self.dim, r))

        # Affine offsets (u_span.span + u_label.label + bias) — applied on top of
        # ANY mode, so "dot + affine" isolates the affine contribution from the
        # bilinear interaction. Init 0 -> starts as a no-op.
        if self.biaffine_affine:
            self.u_span = nn.Parameter(torch.zeros(self.dim))
            self.u_label = nn.Parameter(torch.zeros(self.dim))
            self.affine_bias = nn.Parameter(torch.zeros(1))

        if self.boundary_bias:
            # bias_width: bounded lookup table over span width (K axis).
            self.bias_width = nn.Embedding(self.max_width + 1, 1)
            nn.init.zeros_(self.bias_width.weight)
            # bias_start: fixed sinusoidal encoding (no table -> any L) + a
            # learned linear read-out, init 0 -> no-op until trained.
            self.start_proj = nn.Linear(self.boundary_pos_dim, 1, bias=False)
            nn.init.zeros_(self.start_proj.weight)

        if self.boundary_token:
            # Label-conditioned boundary-token read-outs. Init 0 -> no-op at
            # start (warm-start safe), only opens capacity when trained.
            self.u_start = nn.Parameter(torch.zeros(self.dim))
            self.u_end = nn.Parameter(torch.zeros(self.dim))

        if self.null_score:
            # Learnable label-conditioned null anchor. u_null init 0 and
            # null_bias init logit(null_init_p) -> at step 0, sigmoid(adjusted)
            # > 0.5 is EXACTLY the legacy sigmoid(raw) > null_init_p rule.
            p = min(max(self.null_init_p, 1e-4), 1 - 1e-4)
            self.u_null = nn.Parameter(torch.zeros(self.dim))
            self.null_bias = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))

        if learn_temperature or score_mode == "cosine":
            # temperature scales logits before sigmoid; init exp(temp_init).
            self.log_temp = nn.Parameter(torch.tensor(float(temp_init)))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, mode={self.score_mode}, rank={self.biaffine_rank}, "
            f"residual={self.biaffine_residual}, affine={self.biaffine_affine}, "
            f"lams_rank={self.lams_rank}, boundary_bias={self.boundary_bias}, "
            f"boundary_token={self.boundary_token}, null_score={self.null_score}, "
            f"temp={hasattr(self, 'log_temp')}"
        )

    def _sinusoidal_start_encoding(self, L: int, device, dtype) -> torch.Tensor:
        """Fixed (non-learned) sinusoidal encoding of start index in [0, L).

        No fixed-length table -> works for any ``L`` (768 or 2048 word-chunks),
        unlike an ``nn.Embedding`` which would need a hardcoded max_len.
        """
        d = self.boundary_pos_dim
        position = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d)
        )
        pe = torch.zeros(L, d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.to(dtype)

    def forward(
        self,
        span_rep: torch.Tensor,
        struct_proj: torch.Tensor,
        token_embs: torch.Tensor = None,
    ) -> torch.Tensor:
        mode = self.score_mode

        if mode == "cosine":
            s = F.normalize(span_rep, dim=-1)
            e = F.normalize(struct_proj, dim=-1)
            logits = torch.einsum("lkd,bpd->bplk", s, e)

        elif mode == "biaffine":
            if self.biaffine_rank > 0:
                u = torch.einsum("lkd,dr->lkr", span_rep, self.A)
                v = torch.einsum("bpd,dr->bpr", struct_proj, self.B)   # label side
                logits = torch.einsum("lkr,bpr->bplk", u, v)
                if self.biaffine_residual:
                    logits = logits + torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)
            else:
                lab = torch.einsum("de,bpe->bpd", self.W, struct_proj)  # apply W to label
                logits = torch.einsum("lkd,bpd->bplk", span_rep, lab)

        elif mode == "lams":
            dot = torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)
            u = torch.einsum("lkd,dr->lkr", span_rep, self.U)              # (L,K,r)
            v = torch.einsum("bpd,dr->bpr", struct_proj, self.V)           # (B,P,r) label side
            gate = torch.tanh(torch.einsum("bpd,dr->bpr", struct_proj, self.W_g))  # (B,P,r)
            gated_v = gate * v                                             # (B,P,r)
            logits = dot + torch.einsum("lkr,bpr->bplk", u, gated_v)

        else:  # dot — bit-identical to the original einsum
            logits = torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)

        # Affine offsets — added on top of any mode (dot/cosine/biaffine/lams).
        if self.biaffine_affine:
            span_term = torch.einsum("lkd,d->lk", span_rep, self.u_span)        # (L,K)
            label_term = torch.einsum("bpd,d->bp", struct_proj, self.u_label)   # (B,P)
            logits = (
                logits
                + span_term.unsqueeze(0).unsqueeze(0)      # broadcast over (B,P)
                + label_term.unsqueeze(-1).unsqueeze(-1)   # broadcast over (L,K)
                + self.affine_bias
            )

        # Boundary-position bias — composable with any mode above.
        if self.boundary_bias:
            L, K = span_rep.shape[0], span_rep.shape[1]
            width_idx = torch.arange(min(K, self.max_width + 1), device=span_rep.device)
            w_bias = torch.zeros(K, device=span_rep.device, dtype=logits.dtype)
            w_bias[: width_idx.shape[0]] = self.bias_width(width_idx).squeeze(-1).to(logits.dtype)
            pos_enc = self._sinusoidal_start_encoding(L, span_rep.device, self.start_proj.weight.dtype)
            s_bias = self.start_proj(pos_enc).squeeze(-1).to(logits.dtype)  # (L,)
            boundary = s_bias.unsqueeze(1) + w_bias.unsqueeze(0)            # (L,K)
            logits = logits + boundary.unsqueeze(0).unsqueeze(0)           # broadcast over (B,P)

        # Boundary-token terms — label-conditioned read-out of the span's first
        # and last word embeddings. Composable with any mode above.
        if self.boundary_token:
            if token_embs is None:
                raise RuntimeError(
                    "SpanScorer(boundary_token=True) needs token_embs — the caller "
                    "must thread word-level embeddings (model passes span_info['token_embs'])."
                )
            if token_embs.shape[-1] != self.dim:
                raise RuntimeError(
                    f"boundary_token requires token dim == scorer dim "
                    f"({token_embs.shape[-1]} != {self.dim}); use the uni arch or "
                    f"proj_dim=hidden."
                )
            L, K = span_rep.shape[0], span_rep.shape[1]
            T = token_embs[:L].to(struct_proj.dtype)                        # (L,D)
            # start term: u_start . (h[start] * label) -> (B,P,L), broadcast over K
            start_term = torch.einsum("d,ld,bpd->bpl", self.u_start, T, struct_proj)
            # end term: last word of span (start, width) is index start+width
            # (end-inclusive convention: width = end - start, width 0 = single word)
            idx = (
                torch.arange(L, device=span_rep.device).unsqueeze(1)
                + torch.arange(K, device=span_rep.device).unsqueeze(0)
            ).clamp(max=L - 1)                                              # (L,K)
            T_end = T[idx]                                                  # (L,K,D)
            end_term = torch.einsum("d,lkd,bpd->bplk", self.u_end, T_end, struct_proj)
            logits = logits + start_term.unsqueeze(-1) + end_term

        if hasattr(self, "log_temp"):
            logits = logits * torch.exp(self.log_temp)

        # Null anchor — subtracted LAST so it lives in final logit units.
        # sigmoid(logits) > 0.5  <=>  raw score beats the label's learned null.
        if self.null_score:
            null = torch.einsum("bpd,d->bp", struct_proj, self.u_null) + self.null_bias
            logits = logits - null.unsqueeze(-1).unsqueeze(-1)
        return logits



class CompileSafeGRU(nn.Module):
    """Drop-in for single-layer ``nn.GRU`` that torch.compile can trace.

    Uses the same parameter names (``weight_ih_l0``, ``weight_hh_l0``,
    ``bias_ih_l0``, ``bias_hh_l0``) so pretrained checkpoints that were
    saved with ``nn.GRU`` load without any key remapping.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(3 * hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Run GRU over a sequence.

        Args:
            x: (seq_len, batch, input_size) input sequence.
            h: (batch, hidden_size) initial hidden state.

        Returns:
            (seq_len, batch, hidden_size) — hidden state at each step.
        """
        seq_len = x.shape[0]
        if seq_len == 0:
            return x.new_empty(0, h.shape[0], self.hidden_size)

        outputs = []
        for t in range(seq_len):
            gi = F.linear(x[t], self.weight_ih_l0, self.bias_ih_l0)
            gh = F.linear(h, self.weight_hh_l0, self.bias_hh_l0)

            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)

            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)

            h = (1 - z) * n + z * h
            outputs.append(h)

        return torch.stack(outputs, dim=0)


def create_mlp(input_dim, intermediate_dims, output_dim, dropout=0.1, activation="gelu", add_layer_norm=False):
    """
    Creates a multi-layer perceptron (MLP) with specified dimensions and activation functions.
    """
    activation_mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU
    }
    layers = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers.append(nn.Linear(in_dim, dim))
        if add_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(activation_mapping[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class DownscaledTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initializes a downscaled transformer with specified parameters.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.in_projector = nn.Linear(input_size, hidden_size)

        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

        self.out_projector = create_mlp(
            input_dim=hidden_size + input_size,
            intermediate_dims=[input_size, input_size],
            output_dim=input_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (L, M, input_size).
        Returns:
            Tensor: Output tensor of shape (L, M, input_size).
        """
        original_x = x
        # Project input to hidden size.
        x = self.in_projector(x)
        # Apply transformer encoder.xx
        x = self.transformer(x)
        # Concatenate original input with transformer output.
        x = torch.cat([x, original_x], dim=-1)
        # Project back to input size.
        x = self.out_projector(x)
        return x


class CountLSTM(nn.Module):
    def __init__(self, hidden_size, max_count=20):
        """
        Initializes the module with a learned positional embedding for count steps and a GRU.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        # Learned positional embeddings for count steps: shape (max_count, hidden_size)
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        # Compile-safe GRU (same parameter names as nn.GRU for checkpoint compat)
        self.gru = CompileSafeGRU(input_size=hidden_size, hidden_size=hidden_size)
        # Projector layer: combines GRU output with original embeddings.
        self.projector = create_mlp(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )

    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        """
        Args:
            pc_emb (Tensor): Field embeddings of shape (M, hidden_size).
            gold_count_val (int): Predicted count value (number of steps).
        Returns:
            Tensor: Count-aware structure embeddings of shape (gold_count_val, M, hidden_size).
        """
        M, D = pc_emb.shape
        # Cap gold_count_val by max_count.
        gold_count_val = min(gold_count_val, self.max_count)
        # Create a sequence of count indices: shape (gold_count_val,)
        count_indices = torch.arange(gold_count_val, device=pc_emb.device)
        # Get positional embeddings for each count: (gold_count_val, hidden_size)
        pos_seq = self.pos_embedding(count_indices)
        # Expand pos_seq over the batch dimension: (gold_count_val, M, hidden_size)
        pos_seq = pos_seq.unsqueeze(1).expand(gold_count_val, M, D)
        # Run compile-safe GRU.
        output = self.gru(pos_seq, pc_emb)
        # Concatenate the GRU outputs with the original field embeddings.
        return self.projector(torch.cat([output, pc_emb.unsqueeze(0).expand_as(output)], dim=-1))


class CountLSTMv2(nn.Module):
    def __init__(self, hidden_size, max_count=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
        self.transformer = DownscaledTransformer(
            hidden_size,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

    # NOTE: gold_count_val is now a 0-D Tensor, not a Python int
    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        M, D = pc_emb.size()  # symbolic sizes

        # clamp without dropping to Python
        gold_count_val = min(gold_count_val, self.max_count)

        # build the *full* index vector once, then slice – ONNX supports both ops
        full_idx = torch.arange(self.max_count, device=pc_emb.device)
        count_idx = full_idx[:gold_count_val]  # (gold_count_val,)

        pos_seq = self.pos_embedding(count_idx)  # (gold_count_val, D)
        pos_seq = pos_seq.unsqueeze(1).expand(-1, M, -1)  # (gold_count_val, M, D)

        # Compile-safe GRU forward.
        output = self.gru(pos_seq, pc_emb)  # (gold_count_val, M, D)

        pc_broadcast = pc_emb.unsqueeze(0).expand_as(output)
        return self.transformer(output + pc_broadcast)


class CountLSTMoE(nn.Module):
    """
    Count-aware module with a Mixture-of-Experts projector.

    Args
    ----
    hidden_size : int
        Model dimensionality (D).
    max_count   : int
        Maximum # count steps L.
    n_experts   : int, optional
        Number of FFN experts (default = 4).
    ffn_mult    : int, optional
        Expansion factor for expert bottleneck (default = 2 → inner = 2 D).
    dropout     : float, optional
        Drop-out used inside expert FFNs.
    """

    def __init__(self,
                 hidden_size: int,
                 max_count: int = 20,
                 n_experts: int = 4,
                 ffn_mult: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size, self.max_count, self.n_experts = (
            hidden_size, max_count, n_experts)

        # ───── positional encoding + recurrent core ─────
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)

        # ───── expert parameters (all packed in two tensors) ─────
        inner = hidden_size * ffn_mult
        # W1 : [E, D, inner]    b1 : [E, inner]
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden_size, inner))
        self.b1 = nn.Parameter(torch.zeros(n_experts, inner))
        # W2 : [E, inner, D]  b2 : [E, D]
        self.w2 = nn.Parameter(torch.empty(n_experts, inner, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(n_experts, hidden_size))

        # better than default init for large mats
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        self.dropout = nn.Dropout(dropout)

        # ───── router / gating network ─────
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_experts),  # logits
            nn.Softmax(dim=-1),  # gates sum-to-1
        )

    # ---------------------------------------------------
    def forward(self, pc_emb: torch.Tensor, gold_count_val: int) -> torch.Tensor:
        """
        pc_emb : [M, D]    field embeddings
        gold_count_val : int   (# count steps to unroll)
        returns : [L, M, D]    count-aware embeddings
        """
        M, D = pc_emb.shape
        L = min(gold_count_val, self.max_count)

        idx = torch.arange(L, device=pc_emb.device)
        pos_seq = self.pos_embedding(idx).unsqueeze(1).expand(L, M, D)

        # Compile-safe GRU forward.
        h = self.gru(pos_seq, pc_emb)  # [L, M, D]

        # ───── routing / gating ─────
        gates = self.router(h)  # [L, M, E]

        # ───── expert FFN: run *all* experts in parallel ─────
        # 1st linear
        x = torch.einsum('lmd,edh->lmeh', h, self.w1) + self.b1  # [L, M, E, inner]
        x = F.gelu(x)
        x = self.dropout(x)
        # 2nd linear
        x = torch.einsum('lmeh,ehd->lmed', x, self.w2) + self.b2  # [L, M, E, D]

        # ───── mixture weighted by gates ─────
        out = (gates.unsqueeze(-1) * x).sum(dim=2)  # [L, M, D]
        return out
