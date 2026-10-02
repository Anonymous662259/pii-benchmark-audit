"""Experiment-specific configuration overrides on top of DEFAULT_TRAINING.

Each experiment is a dict that gets merged over DEFAULT_TRAINING.
Import ``EXPERIMENTS`` and access by name::

    from config.experiments import EXPERIMENTS
    cfg = {**DEFAULT_TRAINING, **EXPERIMENTS["exp1_oversample"]}

Experiment IDs
--------------
baseline          — unchanged DEFAULT_TRAINING (nothing overridden)
exp1_oversample   — document-level oversampling (data layer only)
exp2_entity_aug   — entity-aware value replacement augmentation
exp3_lora         — LoRA parameter-efficient fine-tuning
exp4_conservative — conservative encoder LR + cosine schedule + layer freezing
exp5_combined     — LoRA + oversampling + entity-aug + macro-F1 checkpoint
"""

# ── Experiment 1: Oversampling ────────────────────────────────────────────────
# Pure data-pipeline change — training hyperparams unchanged.
# Oversampling config lives here for traceability.
EXP1_OVERSAMPLE = dict(
    # Hyperparams identical to baseline (training is unaffected)
    # Oversampling knobs (read by train.py --experiment exp1_oversample):
    _oversample=True,
    _oversample_rare_threshold=0.25,   # label freq < 25% of mean → rare
    _oversample_max_factor=4.0,        # max repeat multiplier
)

# ── Experiment 2: Entity-aware augmentation ───────────────────────────────────
EXP2_ENTITY_AUG = dict(
    _entity_aug=True,
    _entity_aug_ratio=0.4,    # fraction of eligible docs to augment
    _entity_aug_factor=1,     # augmented copies per selected doc
)

# ── Experiment 3: LoRA fine-tuning ────────────────────────────────────────────
EXP3_LORA = dict(
    use_lora=True,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
    save_adapter_only=True,
    # Slightly higher task_lr is safe because encoder is mostly frozen via LoRA
    encoder_lr=1e-5,
    task_lr=5e-4,
)

# ── Experiment 4: Conservative encoder fine-tuning ───────────────────────────
EXP4_CONSERVATIVE = dict(
    encoder_lr=3e-6,           # 3× lower than baseline
    task_lr=3e-4,             # 1.67× lower than baseline
    # warmup_ratio=0.1,
    # scheduler_type="cosine",
    # # Layer-freezing knobs (read by train.py; not part of TrainingConfig):
    # _freeze_embeddings=True,   # freeze token/position embeddings
    # _freeze_encoder_layers=4,  # freeze bottom N transformer layers
)

# ── Experiment 5: Combined strategy ──────────────────────────────────────────
EXP5_COMBINED = dict(
    # LoRA
    use_lora=False,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
    save_adapter_only=True,
    # Oversampling (via _* private keys)
    _oversample=True,
    _oversample_rare_threshold=0.25,
    _oversample_max_factor=4.0,
    # Entity augmentation
    _entity_aug=True,
    _entity_aug_ratio=0.4,
    _entity_aug_factor=1,
    # Checkpoint selection on macro-F1 instead of eval_loss
    metric_for_best="eval_macro_f1",
    greater_is_better=True,
    # Early stopping
    early_stopping=True,
    early_stopping_patience=2,
)

# ── Experiment 6: Value diversity ────────────────────────────────────────────
# Targets labels with low unique-to-total ratio (PREFIX, GENDER, MARITAL, etc.)
# instead of rare labels.  Three sub-strategies:
#   - diversify: replace over-repeated values with alternatives
#   - mask: create copies with entity values masked to force context learning
#   - cap: drop documents that only contribute over-represented values
EXP6_DIVERSITY = dict(
    _value_diversify=True,
    _value_diversify_max_repeats=5,     # max times any single value may appear
    _value_diversify_threshold=0.15,    # unique/total ratio below this → targeted
    _value_mask=True,
    _value_mask_ratio=0.2,              # fraction of docs to create masked copies
    _value_cap=False,                   # disabled by default (aggressive)
    _value_cap_max_repeats=8,
)

# ── Experiment 7: Combined + diversity ───────────────────────────────────────
# Exp 5 (LoRA + oversample + entity-aug) plus value diversification.
EXP7_COMBINED_DIVERSITY = dict(
    # LoRA
    use_lora=False,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
    save_adapter_only=True,
    # Oversampling
    _oversample=True,
    _oversample_rare_threshold=0.25,
    _oversample_max_factor=4.0,
    # Entity augmentation
    _entity_aug=False,
    _entity_aug_ratio=0.4,
    _entity_aug_factor=1,
    # Value diversity (on top of rare-label strategies)
    _value_diversify=True,
    _value_diversify_max_repeats=5,
    _value_diversify_threshold=0.15,
    _value_mask=True,
    _value_mask_ratio=0.15,
    # Checkpoint
    metric_for_best="eval_macro_f1",
    greater_is_better=True,
    early_stopping=True,
    early_stopping_patience=2,
)

# ── Experiment 8: Class Weight ───────────────────────────────────────────────
# Scalar pos_weight + global per-label inverse-frequency weighting computed
# over the entire training set.  Optionally override weights for specific
# labels that are rare but unimportant (or important but common).
EXP8_CLASS_WEIGHT = dict(
    _loss_pos_weight=1.0,
    _loss_class_weight=True,
    _loss_max_class_weight=2.0,
    _loss_label_weight_overrides={"GENDER": 0.05, "MARITAL": 0.1, "PREFIX": 0.1, "RELIGION": 0.3, "AGE": 0.1},
)

# ── Experiment 9: Label Smoothing ────────────────────────────────────────────
# Smooth targets to prevent overconfident predictions.
# y_smooth = y * (1 - ε) + ε/2.  Helps generalisation on noisy/diverse data.
EXP9_LABEL_SMOOTHING = dict(
    _loss_label_smoothing=0.05,
)

# ── Experiment 10: Oversampling + Class Weight + Label Smoothing ─────────────
# Combined strategy: light oversampling (max 3×) + class weight + LS 0.05.
EXP10_COMBINED_CW_LS = dict(
    # Light oversampling
    _oversample=True,
    _oversample_rare_threshold=0.25,
    _oversample_max_factor=3.0,     # gentler than exp1's 4×
    # Class weight (global, with optional overrides)
    _loss_pos_weight=2.0,
    _loss_class_weight=True,
    _loss_max_class_weight=10.0,
    _loss_label_weight_overrides=None,
    # Label smoothing
)

# ── Experiment 11: Focal Loss ────────────────────────────────────────────────
# Replace BCE with focal loss: FL = -α_t · (1-p_t)^γ · log(p_t).
# Down-weights easy negatives (the vast majority of spans) so the model
# focuses on hard positives.  γ=2 is the RetinaNet default; α=0.25 balances
# positive vs negative class.
EXP11_FOCAL_LOSS = dict(
    _loss_focal=True,
    _loss_focal_gamma=2.0,         # focusing parameter (0 = standard BCE)
    _loss_focal_alpha=0.25,        # positive-class balance factor
)

# ── Experiment 12: Entity IoU Loss (Liu et al., Neurocomputing 2022) ─────────
# Dynamically scaled cross-entropy that up-weights hard negative spans (those
# overlapping with gold entities) via their ENIoU score.
# For positive spans:  L = -(1-α)·log(p)
# For negative spans:  L = -α·(1+ENIoU)·log(1-p)
# α < 0.5 pays more attention to positives (rarer).  Paper default: α=0.25.
EXP12_ENTITY_IOU_LOSS = dict(
    _loss_entity_iou=True,
    _loss_entity_iou_alpha=0.25,   # balance factor (paper default)
)

# ── Experiment 13: Boundary Smoothing (Zhu & Li, ACL 2022) ───────────────────
# Spread probability ε from each gold span to candidate spans near its boundary
# (Manhattan distance ≤ D on the start/end plane) instead of smoothing every
# target uniformly.  Mitigates over-confidence on ambiguous entity boundaries.
# BCE-only (the paper's setting); mutually exclusive with _loss_label_smoothing.
EXP13_BOUNDARY_SMOOTHING = dict(
    _loss_boundary_smoothing=0.1,      # ε, paper range {0.1, 0.2, 0.3}
    _loss_boundary_smoothing_size=1,   # D, paper range {1, 2}
)

# ── Experiment 14: Focal + Entity-IoU combo ──────────────────────────────────
# Convex combination mix·focal + (1-mix)·eiou.  `mix` is the COMBINATION weight
# (not the per-term α); per-term α stays at the paper default 0.25.  The two halves
# live on different magnitudes, so mix=0.5 is not "equal contribution" — tune it.
EXP14_FOCAL_EIOU = dict(
    _loss_focal_eiou=True,
    _loss_mix=0.5,                  # combination weight (mix·focal + (1-mix)·eiou)
    _loss_focal_gamma=2.0,          # focal focusing parameter
    _loss_focal_alpha=0.25,         # focal per-term α
    _loss_entity_iou_alpha=0.25,    # eiou per-term α
)

# ── Experiment 16: Set-Consistent Training ───────────────────────────────────
# Train against the evaluator's set-matching view: every cell above the decision
# boundary (logit > 0 — the pool T of the paper, no cardinality parameter) is
# assigned to golds by max-IoU ownership with the LENIENT IoU criterion
# (ownership hinge + set-level FP mining + cross-label hinge), plus a learnable
# per-label null anchor in the scorer (decode = "beats null" at --threshold 0.5,
# init == legacy 0.7 rule).
# Pair with --affine-exp exp13_biaffine_boundary_token for the full design.
EXP16_SET_CONSISTENT = dict(
    _loss_set=True,
    _loss_set_margin=0.5,
    _loss_set_lambda=1.0,
    _loss_set_lambda_fp=0.25,
    _loss_set_lambda_label=0.25,
    _loss_set_topk=0,               # 0 = pool is {s > 0} (published objective)
    _loss_set_null_init=0.7,
    _loss_boundary_smoothing=0.1,
    _loss_boundary_smoothing_size=1,
)

# ── Experiment 16b: Set-Consistent with the legacy top-K pool ────────────────
# Identical to exp16 except the competition pool is capped at the K=16 highest
# cells instead of {s > 0}.  Kept to reproduce the pre-Eq.5 runs and as the
# pool ablation: under a top-K pool a rival that displaces a gold span may sit
# outside the pool, so the decoding-error decomposition no longer holds.
EXP16B_SET_CONSISTENT_TOPK = dict(EXP16_SET_CONSISTENT, _loss_set_topk=16)

# ── Registry ──────────────────────────────────────────────────────────────────
EXPERIMENTS: dict = {
    "baseline":         {},                   # no overrides
    "exp1_oversample":  EXP1_OVERSAMPLE,
    "exp2_entity_aug":  EXP2_ENTITY_AUG,
    "exp3_lora":        EXP3_LORA,
    "exp4_conservative": EXP4_CONSERVATIVE,
    "exp5_combined":    EXP5_COMBINED,
    "exp6_diversity":   EXP6_DIVERSITY,
    "exp7_combined_diversity": EXP7_COMBINED_DIVERSITY,
    "exp8_class_weight": EXP8_CLASS_WEIGHT,
    "exp9_label_smoothing": EXP9_LABEL_SMOOTHING,
    "exp10_combined_cw_ls": EXP10_COMBINED_CW_LS,
    "exp11_focal_loss": EXP11_FOCAL_LOSS,
    "exp12_entity_iou_loss": EXP12_ENTITY_IOU_LOSS,
    "exp13_boundary_smoothing": EXP13_BOUNDARY_SMOOTHING,
    "exp14_focal_eiou": EXP14_FOCAL_EIOU,
    "exp16_set_consistent": EXP16_SET_CONSISTENT,
    "exp16b_set_consistent_topk": EXP16B_SET_CONSISTENT_TOPK,
}

# Private keys (prefixed with _) are consumed by the training script and must
# NOT be passed to TrainingConfig.
_PRIVATE_KEYS = frozenset({
    "_oversample", "_oversample_rare_threshold", "_oversample_max_factor",
    "_entity_aug", "_entity_aug_ratio", "_entity_aug_factor",
    "_freeze_embeddings", "_freeze_encoder_layers",
    "_value_diversify", "_value_diversify_max_repeats", "_value_diversify_threshold",
    "_value_mask", "_value_mask_ratio",
    "_value_cap", "_value_cap_max_repeats",
    "_loss_pos_weight", "_loss_class_weight", "_loss_max_class_weight",
    "_loss_label_smoothing", "_loss_label_weight_overrides",
    "_loss_focal", "_loss_focal_gamma", "_loss_focal_alpha",
    "_loss_entity_iou", "_loss_entity_iou_alpha",
    "_loss_boundary_smoothing", "_loss_boundary_smoothing_size",
    "_loss_focal_eiou", "_loss_mix",
    "_loss_set", "_loss_set_margin", "_loss_set_lambda", "_loss_set_lambda_fp",
    "_loss_set_lambda_label", "_loss_set_topk", "_loss_set_null_init", "_loss_set_base_margin",
})


def split_experiment_config(experiment_name: str, base: dict) -> tuple:
    """Return (training_cfg_dict, private_opts_dict).

    Merges the experiment overrides onto ``base`` (usually DEFAULT_TRAINING),
    then separates TrainingConfig kwargs from private pipeline knobs.
    """
    overrides = EXPERIMENTS.get(experiment_name, {})
    merged = {**base, **overrides}
    private = {k: merged.pop(k) for k in list(merged) if k in _PRIVATE_KEYS}
    return merged, private
