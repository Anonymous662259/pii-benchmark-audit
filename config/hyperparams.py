"""Default hyperparameters shared across training experiments."""

import os
from pathlib import Path


# ── HuggingFace org & auth ────────────────────────────────────────────────────
def _read_hf_token() -> str:
    """Resolve the HF token: ``HF_TOKEN`` env var, else a ``.hf_token`` file at
    the repo root (gitignored).

    NEVER hardcode a token in this module.  It is tracked by git, so a literal
    token goes into history permanently and rotating it is then the only fix.
    On Modal the repo is synced into a Volume, so a ``.hf_token`` file travels
    with it and works without wiring a Secret through the runner.
    """
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    p = Path(__file__).resolve().parent.parent / ".hf_token"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return ""


HF_TOKEN = _read_hf_token()
HF_ORG   = "anonymous-org"

# Collection slugs — fill in after creating collections on HF Hub UI:
#   https://huggingface.co/anonymous-org → Collections → New collection
MODEL_COLLECTION_SLUG = "anonymous-org/securepii-model"  # e.g. "anonymous-org/securepii-<id>"
BENCH_COLLECTION_SLUG ="anonymous-org/securepii-bench"   # e.g. "anonymous-org/securepii-bench-<id>"

# ── Dataset ──────────────────────────────────────────────────────────────────
TRAIN_DATASET_ID  = "anonymous-org/ontonotes5"
EVAL_DATASET_ID   = "anonymous-org/ontonotes5"
EVAL_SPLIT        = "test"
LANGUAGES         = ["en", "de", "vi"]
MAX_TRAIN_SAMPLES = 100000
MAX_VAL_SAMPLES   = 50000

# ── K-Fold Cross-Validation ──────────────────────────────────────────────────
FOLD_DATASET_ID = "anonymous-org/rosenxt-v9-25-6"   # dataset with fold_k_train / fold_k_test splits
N_FOLDS         = 5

# Fine-tune validation set (secondary eval source)
FINETUNE_VAL_DATASET  = "anonymous-org/rosenxt-combine-ne-v2"
FINETUNE_VAL_SPLIT    = "validation"
FINETUNE_VAL_MAX      = 50000
EXTRA_LANGS_FROM_FT   = ["en", "de", "vi"]
USE_EXTRA_FT_VAL      = True

# ── Training defaults (Baseline & DataAug) ───────────────────────────────────
USE_LABEL_DESCRIPTIONS = True
MAX_WIDTH              = 30

DEFAULT_TRAINING = dict(
    num_epochs=3,
    max_steps=-1,
    batch_size=8,
    eval_batch_size=16,
    max_len=2048,          # word-level tokens (WhitespaceTokenSplitter units)
    encoder_lr=1e-5,
    task_lr=5e-4,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    scheduler_type="linear",
    warmup_ratio=0.1,
    warmup_steps=0,
    num_cycles=0.5,
    fp16=True,
    bf16=False,
    eval_strategy="epoch",
    save_total_limit=2,
    save_best=True,
    metric_for_best="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    logging_first_step=True,
    report_to_wandb=False,
    early_stopping=False,
    early_stopping_patience=3,
    early_stopping_threshold=0.0,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    seed=42,
    deterministic=False,
    validate_data=True,
    use_lora=False,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.0,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
    save_adapter_only=True,
)

# ── mmBERT overrides (applied on top of DEFAULT_TRAINING) ───────────────────
MMBERT_OVERRIDES = dict(
    num_epochs=6,         # cold-start task heads need more epochs
    max_len=2048,
    task_lr=2e-4,
    adam_epsilon=1e-6,
    fp16=False,
    bf16=True,            # H100 native bfloat16
    eval_strategy="epoch",
    early_stopping=True,
    early_stopping_patience=2,
)

# ── Data augmentation defaults ───────────────────────────────────────────────
AUG_FACTOR = 1     # augmented copies per selected example
AUG_RATIO  = 0.5   # fraction of train examples to augment
AUG_SEED   = 42

# ── Evaluation defaults ──────────────────────────────────────────────────────
THRESHOLD              = 0.7
BENCHMARK_THRESHOLD    = 0.7   # stricter threshold used in full benchmark runs

# Per-label threshold overrides for schema building.
# Labels listed here use the given threshold; all others use the global
# threshold passed to model.extract() (BENCHMARK_THRESHOLD = 0.7).
PER_LABEL_THRESHOLDS = {
}
IOU_THRESH             = 0.5   # lenient/fuzzy scoring (default HungarianEvaluator config)
FUZZY_MAX_DIST         = 3     # chars of start-offset slack allowed by the lenient fuzzy match
IOU_THRESH_STRICT      = 1.0   # exact-match scoring (span boundaries must line up exactly)
FUZZY_MAX_DIST_STRICT  = None  # None disables the fuzzy fallback entirely (pure IoU==1.0 match);
                                # 0 would NOT be exact — the substring check still lets a
                                # truncated/extended prediction match if its start lines up.
TIMING_PROBE_SAMPLES   = 100   # sequential samples timed for samples/s before switching to batch inference
MAX_TOKENS_PER_CHUNK   = 2048
MAX_TOKENS_PER_CHUNK   = 2048
CHUNK_SAFETY_MARGIN    = 20
MAX_CHARS_PER_CHUNK    = 4048
RANDOM_SEED            = 123
PARALLEL_BATCH_SIZE    = 8

# ── Benchmark model registry ─────────────────────────────────────────────────
BENCHMARK_MODELS = [
    {
        "name":         "glinerv2",
        "display":      "glinerv2",
        "repo_id":      "anonymous-org/gliner2-multi-v1-3e-20260514",
        "subfolder":    None,
        "max_width":    30,
        "chunk":        True,
        "chunk_tokens": 768,
    },
    {
        "name":         "glinerv2_data_aug",
        "display":      "glinerv2 data aug",
        "repo_id":      "anonymous-org/gliner2-multi-v1-3e-20260514-dataaug",
        "subfolder":    None,
        "max_width":    30,
        "chunk":        True,
        "chunk_tokens": 768,
    },
    {
        "name":         "glinerv2_mmbert_backbone",
        "display":      "glinerv2 mmbert backbone",
        "repo_id":      "anonymous-org/gliner2-mmbert-base-rosenxt-6e-20260515",
        "subfolder":    None,
        "max_width":    30,
        "chunk":        True,
        "chunk_tokens": 2048,
    },
    {
        "name":         "glinerv2_quy",
        "display":      "glinerv2 quy",
        "repo_id":      "anonymous-org/gliner2-rosenxt-full-20260515-quy",
        "subfolder":    "best",
        "max_width":    None,
        "chunk":        False,
        "chunk_tokens": None,
    },
    {
        "name":         "gliner_v2_quy_code_sai",
        "display":      "gliner v2 quy code sai",
        "repo_id":      "anonymous-org/gliner2-rosenxt-multi-v7",
        "subfolder":    "best",
        "max_width":    None,
        "chunk":        False,
        "chunk_tokens": None,
    },
]
