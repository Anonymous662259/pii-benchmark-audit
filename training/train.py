"""CLI training script for GLiNER2 PII extraction experiments.

Usage:
    # Baseline
    python training/train.py --model fastino/gliner2-multi-v1 --repo anonymous-org/my-model

    # With data augmentation (original label-order shuffle)
    python training/train.py --model fastino/gliner2-multi-v1 --augment --repo anonymous-org/my-model-dataaug

    # Named experiments (Exp 1–5 imbalance mitigation)
    python training/train.py --model fastino/gliner2-multi-v1 --experiment exp1_oversample
    python training/train.py --model fastino/gliner2-multi-v1 --experiment exp2_entity_aug
    python training/train.py --model fastino/gliner2-multi-v1 --experiment exp3_lora
    python training/train.py --model fastino/gliner2-multi-v1 --experiment exp4_conservative
    python training/train.py --model fastino/gliner2-multi-v1 --experiment exp5_combined

    # mmBERT backbone
    python training/train.py --model jhu-clsp/mmBERT-base --mmbert --epochs 6 --repo anonymous-org/my-model-mmbert

    # Uni-encoder backbone experiment (cold-start on a registry backbone)
    python training/train.py --backbone mdeberta_base --arch uni
    python training/train.py --backbone mdeberta_gliner2_fastino --dtype bf16 --epochs 6

    # K-fold CV (fold 0): trains on fold_0_train from FOLD_DATASET_ID
    python training/train.py --model fastino/gliner2-multi-v1 --fold 0 --repo anonymous-org/gliner2-fold0-20260526
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gliner2.training.data import TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tqdm import tqdm

from config.hyperparams import (
    BENCH_COLLECTION_SLUG,
    DEFAULT_TRAINING,
    FOLD_DATASET_ID,
    HF_ORG,
    HF_TOKEN,
    MAX_WIDTH,
    MMBERT_OVERRIDES,
    MODEL_COLLECTION_SLUG,
    N_FOLDS,
    USE_LABEL_DESCRIPTIONS,
)
from config.labels import PII_LABEL_DESCRIPTIONS
from config.experiments import EXPERIMENTS, split_experiment_config
from data.augmentation import augment_train_examples
from data.converter import is_valid_example, to_input_example
from data.loader import load_fold_split, load_train_val
from data.oversampling import oversample_rare_labels, print_oversampling_stats
from data.entity_augmentation import entity_aware_augment, print_entity_aug_stats
from data.oversampling import detect_rare_labels, compute_label_frequencies
from data.value_diversifier import (
    detect_low_diversity_labels, diversify_values, mask_entity_values,
    cap_value_occurrences, print_diversity_stats,
)
from models.loader import load_gliner2_model
from models.loss_wrapper import apply_weighted_loss, apply_focal_loss, apply_entity_iou_loss, apply_focal_eiou_loss, apply_margin_loss, apply_set_consistent_loss, compute_global_class_weights
from models.tokenizer_patch import apply_tokenizer_patch
from models.freeze import freeze_embeddings, freeze_encoder_layers, print_frozen_summary


_DTYPE_FLAGS = {
    "bf16": dict(fp16=False, bf16=True),
    "fp16": dict(fp16=True, bf16=False),
    "fp32": dict(fp16=False, bf16=False),
}


# Default projection dim D per bi-family arch: both default to "hidden"
# (D = backbone hidden size, no bottleneck) so bi / two-tower / uni all score in
# the same dim H — a fair joint-vs-separate-encoding comparison. Pass
# --proj-dim 256 to reproduce the hivetrace bi bottleneck.
_ARCH_DEFAULT_PROJ = {"bi": "hidden", "bi-encoder-2-encoder": "hidden"}

# Short repo/dir tag per arch (matches scripts/run_arch_experiments.py).
_ARCH_TAG = {"uni": "uni", "bi": "bi", "bi-encoder-2-encoder": "bi2enc"}


def _proj_tag(args) -> str:
    """Name suffix when D differs from the arch's default, so e.g. D=256 vs
    D=hidden repos/dirs don't collide ('-pdhidden'). Empty for uni or the default."""
    default = _ARCH_DEFAULT_PROJ.get(args.arch)
    if default is not None and str(args.proj_dim) != default:
        return f"-pd{args.proj_dim}"
    return ""


def build_config(args, overrides=None) -> tuple:
    """Return (TrainingConfig, private_opts_dict).

    private_opts_dict contains pipeline knobs prefixed with _ that must NOT be
    forwarded to TrainingConfig (oversampling ratios, freeze layer counts, etc.).
    ``overrides`` (affine_adapt train_overrides) are merged last over params.
    """
    params = dict(DEFAULT_TRAINING)
    if args.mmbert:
        params.update(MMBERT_OVERRIDES)
    if args.backbone:
        # uni-encoder backbone experiment: per-backbone overrides (max_len,
        # cold-start epochs, dtype, lr) from the registry.
        from config.backbones import get_backbone, training_overrides
        params.update(training_overrides(get_backbone(args.backbone)))
    if args.warmstart_gliner2:
        # source encoder is mdeberta-v3 (deberta-v2) → fp32 to avoid disentangled-
        # attention NaN; warm heads → baseline recipe (max_len 1024 like registry).
        params.update(fp16=False, bf16=False, max_len=1024)
    if args.dtype:                       # explicit --dtype wins over registry
        params.update(_DTYPE_FLAGS[args.dtype])

    # Apply named experiment overrides (strips private _ keys)
    if args.experiment and args.experiment != "baseline":
        params, private_opts = split_experiment_config(args.experiment, params)
    else:
        private_opts = {}

    if args.batch_size:                  # lower to fit memory (OOM)
        params["batch_size"] = args.batch_size
        params["eval_batch_size"] = args.batch_size
    if args.grad_accum:                  # keep effective batch = batch_size * grad_accum
        params["gradient_accumulation_steps"] = args.grad_accum
    if args.early_stopping:              # explicit --early-stopping wins over registry/experiment/mmbert
        params["early_stopping"] = True
    if args.early_stopping_patience is not None:
        params["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_threshold is not None:
        params["early_stopping_threshold"] = args.early_stopping_threshold
    params["num_epochs"] = args.epochs or params["num_epochs"]
    if args.seed is not None:            # multi-seed runs: measure run-to-run variance
        params["seed"] = args.seed
    params["output_dir"] = args.output_dir
    if args.repo:
        params["experiment_name"] = args.repo.split("/")[-1]
    elif args.warmstart_gliner2:
        params["experiment_name"] = f"gliner2-warmstart-{_ARCH_TAG.get(args.arch, args.arch)}"
    elif args.backbone:
        params["experiment_name"] = f"gliner2-{args.backbone}-{args.arch}{_proj_tag(args)}"
    elif args.fold is not None:
        params["experiment_name"] = f"gliner2-fold{args.fold}"
    elif args.experiment and args.experiment != "baseline":
        params["experiment_name"] = f"gliner2-{args.experiment}"
    else:
        params["experiment_name"] = "gliner2-run"
    # affine_adapt experiment may override TrainingConfig knobs (epochs, lrs, dtype…)
    if overrides:
        params.update(overrides)
    return TrainingConfig(**params), private_opts


def _merge_cli_loss(args, private_opts: dict) -> None:
    """Fold the orthogonal --loss-* CLI flags into ``private_opts`` (in place).

    CLI takes precedence over the loss bits of --experiment.  Maps onto the same
    ``_loss_*`` keys the loss-application block already consumes, so both paths
    converge on one tested code path.  A no-op when no loss flag was passed.
    """
    touched = (
        args.loss is not None
        or args.loss_class_weight
        or args.pos_weight is not None
        or args.loss_label_smoothing is not None
        or args.loss_gamma is not None
        or args.loss_alpha is not None
        or args.loss_mix is not None
        or args.loss_margin is not None
        or args.set_margin is not None
        or args.set_margin_label is not None
        or args.set_iou_match is not None
        or args.set_lambda is not None
        or args.set_lambda_fp is not None
        or args.set_lambda_label is not None
        or args.set_topk is not None
        or args.set_base_margin is not None
        or args.set_null_init is not None
        or args.boundary_smoothing is not None
        or args.boundary_smoothing_size is not None
    )
    if not touched:
        return

    if args.loss is not None:
        # Switching term: clear the other terms so the if/elif block picks ours.
        private_opts.pop("_loss_focal", None)
        private_opts.pop("_loss_entity_iou", None)
        private_opts.pop("_loss_focal_eiou", None)
        private_opts.pop("_loss_margin", None)
        private_opts.pop("_loss_set", None)
        if args.loss == "focal":
            private_opts["_loss_focal"] = True
            if args.loss_gamma is not None:
                private_opts["_loss_focal_gamma"] = args.loss_gamma
            if args.loss_alpha is not None:
                private_opts["_loss_focal_alpha"] = args.loss_alpha
        elif args.loss == "eiou":
            private_opts["_loss_entity_iou"] = True
            if args.loss_alpha is not None:
                private_opts["_loss_entity_iou_alpha"] = args.loss_alpha
        elif args.loss == "focal_eiou":
            private_opts["_loss_focal_eiou"] = True
            if args.loss_mix is not None:
                private_opts["_loss_mix"] = args.loss_mix
            if args.loss_gamma is not None:
                private_opts["_loss_focal_gamma"] = args.loss_gamma
        elif args.loss == "margin":
            private_opts["_loss_margin"] = True
            if args.loss_margin is not None:
                private_opts["_loss_margin_value"] = args.loss_margin
        elif args.loss == "set":
            private_opts["_loss_set"] = True
            if args.set_margin is not None:
                private_opts["_loss_set_margin"] = args.set_margin
            if args.set_margin_label is not None:
                private_opts["_loss_set_margin_label"] = args.set_margin_label
            if args.set_iou_match is not None:
                private_opts["_loss_set_iou_match"] = args.set_iou_match
            if args.set_lambda is not None:
                private_opts["_loss_set_lambda"] = args.set_lambda
            if args.set_lambda_fp is not None:
                private_opts["_loss_set_lambda_fp"] = args.set_lambda_fp
            if args.set_lambda_label is not None:
                private_opts["_loss_set_lambda_label"] = args.set_lambda_label
            if args.set_topk is not None:
                private_opts["_loss_set_topk"] = args.set_topk
            if args.set_base_margin is not None:
                private_opts["_loss_set_base_margin"] = args.set_base_margin
            if args.set_null_init is not None:
                private_opts["_loss_set_null_init"] = args.set_null_init

    if args.pos_weight is not None:
        private_opts["_loss_pos_weight"] = args.pos_weight
    if args.loss_class_weight:
        private_opts["_loss_class_weight"] = True
    if args.loss_label_smoothing is not None:
        private_opts["_loss_label_smoothing"] = args.loss_label_smoothing
    if args.boundary_smoothing is not None:
        private_opts["_loss_boundary_smoothing"] = args.boundary_smoothing
    if args.boundary_smoothing_size is not None:
        private_opts["_loss_boundary_smoothing_size"] = args.boundary_smoothing_size


def _push_to_hub(local_dir: str, repo_id: str):
    from huggingface_hub import add_collection_item, create_repo, upload_folder
    create_repo(repo_id, repo_type="model", exist_ok=True)
    upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="best checkpoint",
    )
    print(f"Model → https://huggingface.co/{repo_id}")
    if MODEL_COLLECTION_SLUG:
        add_collection_item(MODEL_COLLECTION_SLUG, repo_id, item_type="model", exists_ok=True)
        print(f"Added to collection SecurePII")


def build_parser():
    """The CLI, split out of ``main`` so an experiment harness can assert that a
    flag string maps to the options it intends (see tests/test_ablation_cli.py):
    an ablation whose "-own" row silently keeps the ownership hinge would look
    like a result rather than a bug."""
    from config.backbones import BACKBONES
    parser = argparse.ArgumentParser(description="Train GLiNER2 for PII extraction")
    parser.add_argument("--model", default=None, help="Base model HF repo or local path (omit if --backbone)")
    parser.add_argument(
        "--backbone", default=None, choices=sorted(BACKBONES),
        help="Uni-encoder backbone experiment: cold-start a fresh GLiNER2 on this "
             "registry encoder (config/backbones.py). Mutually exclusive with --model.",
    )
    parser.add_argument(
        "--arch", default="uni", choices=["uni", "bi", "bi-encoder-2-encoder"],
        help="Encoder architecture (default: uni). 'bi' = shared-weight bi-encoder "
             "(one backbone encodes text & labels separately + label caching). "
             "'bi-encoder-2-encoder' = two separate encoders (siamese init, both "
             "trained, diverge during training). Both bi archs require --backbone.",
    )
    parser.add_argument(
        "--dtype", default=None, choices=["bf16", "fp16", "fp32"],
        help="Training precision. Default: let the backbone registry decide "
             "(deberta-v2 → fp32 to avoid NaN; modernbert/mmbert → bf16); "
             "--model path falls back to DEFAULT_TRAINING (fp16). Pass to override.",
    )
    parser.add_argument(
        "--proj-dim", default=None,
        help="Bi-family projection dim D. 'hidden' = backbone hidden size (D=H, "
             "default for both bi and bi-encoder-2-encoder → scores in H like uni) "
             "or an int (e.g. 256 = hivetrace bi bottleneck). Ignored for uni.",
    )
    parser.add_argument("--repo", default=None, help="Output HF repo (defaults to anonymous-org/<name>)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override TrainingConfig.seed (default 42). Use several seeds "
                             "to measure the noise floor before reading a sweep — on small "
                             "sets like FIN one entity can move macro-F1 by ~4 points.")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Cap validation samples used by the in-training eval loop "
                             "(default: config MAX_VAL_SAMPLES=50000). Lower this to speed up "
                             "per-epoch evaluation. Ignored in --fold mode.")
    parser.add_argument("--augment", action="store_true", help="Enable label-order shuffle augmentation")
    parser.add_argument("--mmbert", action="store_true", help="Use CustomExtractor for mmBERT backbone")
    parser.add_argument("--output-dir", default="./output", help="Local checkpoint output dir")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip pushing to HF Hub after training (e.g. smoke tests). "
                             "Without this, --backbone/--fold auto-push to anonymous-org/<name>.")
    parser.add_argument(
        "--fold", type=int, default=None, metavar="K",
        help=f"K-fold CV mode: train on fold_K_train from {FOLD_DATASET_ID} (K in 0..{N_FOLDS - 1}). "
             "Overrides the default rosenxt training dataset.",
    )
    parser.add_argument(
        "--experiment", default="baseline",
        choices=list(EXPERIMENTS.keys()),
        help="Named experiment preset (baseline | exp1_oversample | exp2_entity_aug | "
             "exp3_lora | exp4_conservative | exp5_combined).  Overrides DEFAULT_TRAINING "
             "hyperparams and enables the corresponding data-pipeline options.",
    )
    parser.add_argument(
        "--affine-exp", default=None, metavar="NAME|PATH",
        help="Span-scorer experiment (feat/affine_adapt): name under "
             "experiments/affine/configs/ (e.g. exp01_biaffine_full) or a JSON path. "
             "Swaps the dot scorer for biaffine/cosine and sets the freeze tier.",
    )
    parser.add_argument(
        "--warmstart-gliner2", default=None, metavar="HF_REPO",
        help="Warm-start a bi / bi-encoder-2-encoder model from a native GLiNER2 "
             "checkpoint (heads included, e.g. fastino/gliner2-multi-v1): copies "
             "the encoder + 4 task heads and identity-inits the projections. "
             "Use with --arch bi|bi-encoder-2-encoder; do NOT pass --model/--backbone.",
    )
    parser.add_argument(
        "--dataset", default=None, metavar="HF_REPO",
        help="Training dataset HF repo (default: config TRAIN_DATASET_ID). E.g. "
             "anonymous-org/rosenxt-v8-19-6-full to fine-tune on the set the checkpoints "
             "were trained on. Ignored in --fold mode.",
    )
    parser.add_argument(
        "--desc-file", default=None,
        help="JSON {label: description} overriding the registry descriptions for "
             "--ner-dataset TRAINING (rich/contrastive/exemplar schemas). Keys must "
             "match the registry labels (case-insensitive; registry casing kept, so "
             "gold alignment is untouched). Evaluate the checkpoint with the SAME "
             "--desc-file (evaluation/evaluate.py supports the same flag).",
    )
    parser.add_argument(
        "--ner-dataset", default=None, metavar="KEY",
        help="NER benchmark dataset key (e.g. conll2003, bc5cdr, ontonotes5). "
             "Switches to the NER data pipeline: loads from the registry repo, "
             "uses dataset-native label descriptions (not PII labels), and "
             "carves 5%% of train as in-training val. Compatible with --repo "
             "and all loss/arch flags. Available: "
             "conll2003 bc5cdr fin ontonotes5 mit_restaurant bionlp2004 wnut2017 multinerd.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch_size and eval_batch_size. Lower to fix OOM (e.g. 2).",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None,
        help="gradient_accumulation_steps. Pair with a small --batch-size to keep the "
             "effective batch (e.g. --batch-size 2 --grad-accum 4 == effective 8).",
    )
    # ── Orthogonal loss axis (alternative to --experiment exp8..12) ──────────
    parser.add_argument(
        "--loss", default=None, choices=["bce", "focal", "eiou", "focal_eiou", "margin", "set"],
        help="Structure-loss term (orthogonal to arch/backbone/data). Overrides the "
             "loss bits of --experiment. bce=default; focal/eiou for span imbalance; "
             "focal_eiou = mix·focal + (1-mix)·eiou; margin = additive-margin BCE "
             "(CosFace/ArcFace-style, no added params — separates semantically-close labels); "
             "set = Set-Consistent: match the top-K predicted set against gold by the "
             "evaluator's own lenient-IoU criterion — usurper hinge + set-level FP "
             "mining + learnable null anchor (decode at --threshold 0.5).",
    )
    parser.add_argument("--loss-gamma", type=float, default=None, help="focal γ (default 2.0)")
    parser.add_argument("--loss-alpha", type=float, default=None, help="focal/eiou α (default 0.25)")
    parser.add_argument("--loss-mix", type=float, default=None,
                        help="focal_eiou combination weight (mix·focal + (1-mix)·eiou; default 0.5). "
                             "NOT the per-term α")
    parser.add_argument("--loss-margin", type=float, default=None,
                        help="additive margin m for --loss margin (default 0.15): gold logits "
                             "are shifted by -m before BCE, demanding score(gold) - score(neg) > m")
    parser.add_argument("--set-margin", type=float, default=None,
                        help="set: hinge margin gold-vs-matched-representative (default 0.5)")
    parser.add_argument("--set-margin-label", "--set-label-margin",
                        dest="set_margin_label", type=float, default=None,
                        help="set: cross-label/type hinge margin at gold cells (default 0.5)")
    parser.add_argument("--set-iou-match", type=float, default=None,
                        help="set: IoU ownership threshold tau (default 0.5)")
    parser.add_argument("--set-lambda", type=float, default=None,
                        help="set: weight of the usurper hinge (default 1.0; 0 disables)")
    parser.add_argument("--set-lambda-fp", type=float, default=None,
                        help="set: weight of set-level FP mining (default 0.25; 0 disables)")
    parser.add_argument("--set-lambda-label", type=float, default=None,
                        help="set: weight of the cross-label hinge at gold cells (default 0.25)")
    parser.add_argument("--set-topk", type=int, default=None,
                        help="set: legacy cap on the competition pool (default 0 = the "
                             "published objective, pool = every cell with logit > 0). "
                             "K>0 restricts it to the K highest cells — ablation only, "
                             "it breaks the paper's decoding-error decomposition.")
    parser.add_argument("--set-base-margin", type=float, default=None,
                        help="set: additive margin m applied to the BCE base inside the set "
                             "loss (margin_bce as base instead of plain BCE; default 0 = off). "
                             "Combines the two best-measured components: margin base + "
                             "set-level competition + null anchor.")
    parser.add_argument("--set-null-init", type=float, default=None,
                        help="set: null anchor init probability p — step-0 decode equals the "
                             "legacy fixed threshold p (default 0.7). Evaluate set-trained "
                             "checkpoints with --threshold 0.5.")
    parser.add_argument("--null-anchor", action="store_true",
                        help="Orthogonal scorer axis: enable the learnable label-conditioned "
                             "null anchor with ANY loss (margin/bce/set/...). The scorer "
                             "learns the per-label operating point end-to-end (init = legacy "
                             "threshold --null-init). ⚠️ Evaluate with --threshold 0.5. "
                             "--loss set enables this implicitly.")
    parser.add_argument("--no-null-anchor", action="store_true",
                        help="Suppress the null anchor that --loss set enables implicitly, "
                             "isolating the competition terms from the learned decision "
                             "boundary (the -null row of the ablation). Decode such a "
                             "checkpoint at the legacy --threshold 0.7, not 0.5.")
    parser.add_argument("--full-grid", action="store_true",
                        help="BCE only: turn off the random negative-span masking so the "
                             "base sees every cell, as --loss set always does. Pair with the "
                             "bce_bundle recipe so it differs from set by the competition "
                             "terms alone (PROTOCOL.md).")
    parser.add_argument("--null-init", type=float, default=None,
                        help="Null anchor init probability p for --null-anchor (default 0.7: "
                             "step-0 decode identical to the legacy fixed-0.7 rule).")
    parser.add_argument("--pos-weight", type=float, default=None, help="BCE positive-class weight")
    parser.add_argument("--loss-class-weight", action="store_true",
                        help="Global inverse-freq per-label weighting (works with any --arch now)")
    parser.add_argument("--loss-label-smoothing", type=float, default=None, help="target smoothing ε")
    parser.add_argument("--boundary-smoothing", type=float, default=None,
                        help="boundary smoothing ε (Zhu & Li 2022); bce-only, excludes --loss-label-smoothing")
    parser.add_argument("--boundary-smoothing-size", type=int, default=None,
                        help="boundary smoothing radius D (Manhattan, default 1)")
    # ── Orthogonal early-stopping axis (works with any mode, incl. --ner-dataset) ──
    parser.add_argument("--early-stopping", action="store_true",
                        help="Stop training once eval_loss (or --experiment's metric_for_best) "
                             "stops improving for --early-stopping-patience consecutive evals. "
                             "Off by default (DEFAULT_TRAINING); overrides --experiment/--backbone/--mmbert.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Evals with no improvement before stopping (default: 3, or the "
                             "registry/experiment/--mmbert value if set and this flag is omitted).")
    parser.add_argument("--early-stopping-threshold", type=float, default=None,
                        help="Minimum eval_loss improvement to reset patience (default: 0.0).")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve the per-arch default projection dim (so --proj-dim can be omitted).
    if args.proj_dim is None:
        args.proj_dim = _ARCH_DEFAULT_PROJ.get(args.arch, "256")

    # ── Single validation point for all 7 axes (init/arch/dtype/loss/...) ────
    from config.spec import RunSpec, RunSpecError
    try:
        run_spec = RunSpec.from_args(args)
        for _w in run_spec.validate():
            print(f"[RunSpec][warn] {_w}")
    except RunSpecError as e:
        parser.error(str(e))
    if args.desc_file and not args.ner_dataset:
        parser.error("--desc-file requires --ner-dataset")
    if args.full_grid and args.loss not in (None, "bce", "set"):
        parser.error(f"--full-grid is implemented for the BCE base only (--loss set is "
                     f"full-grid already); it would be a silent no-op with --loss {args.loss}.")

    from huggingface_hub import login
    if HF_TOKEN:
        login(token=HF_TOKEN)
    elif not args.no_push:
        parser.error(
            "no HF token: set the HF_TOKEN env var or write it to .hf_token at "
            "the repo root (gitignored).  Pushing to "
            f"{args.repo or 'anonymous-org/<name>'} needs write access; use "
            "--no-push to train without pushing."
        )

    apply_tokenizer_patch()

    if args.ner_dataset:
        # ── NER benchmark mode ────────────────────────────────────────────────
        from config.ner_datasets import NER_DATASETS, resolve_split
        if args.ner_dataset not in NER_DATASETS:
            parser.error(f"Unknown --ner-dataset '{args.ner_dataset}'. "
                         f"Available: {', '.join(sorted(NER_DATASETS.keys()))}")
        ner_cfg = NER_DATASETS[args.ner_dataset]
        ner_label_descriptions = ner_cfg["label_descriptions"]
        if args.desc_file:
            from config.ner_datasets import load_desc_override
            ner_label_descriptions = load_desc_override(args.desc_file, ner_label_descriptions)
            print(f"[NER] Descriptions overridden from {args.desc_file} — evaluate "
                  f"this checkpoint with the SAME --desc-file")
        ner_repo = args.dataset or ner_cfg["repo"]

        from datasets import get_dataset_split_names, load_dataset as _hf_load
        from data.normalizer import normalize_record_generic
        from data.converter import to_input_example_generic
        import random as _rng

        try:
            available_splits = set(get_dataset_split_names(ner_repo))
        except Exception as exc:
            parser.error(f"--ner-dataset {args.ner_dataset}: could not list splits "
                         f"for {ner_repo}: {exc}")

        if ner_cfg["train_split"] not in available_splits:
            parser.error(
                f"--ner-dataset {args.ner_dataset}: repo {ner_repo} has no "
                f"'{ner_cfg['train_split']}' split (available: {sorted(available_splits)}); "
                f"cannot fine-tune."
            )

        # Use the repo's own val split only when the repo has all three of
        # train/val/test natively; otherwise fall back to carving 5% off train
        # (keeps train.py from silently reusing the same split evaluate.py
        # might fall back to as "test").
        val_split = resolve_split(available_splits, ner_cfg.get("val_splits", []))
        test_split = resolve_split(available_splits, ner_cfg.get("test_splits", []))
        use_native_val = val_split is not None and test_split is not None

        print(f"[NER] Dataset  : {ner_repo} (train: {ner_cfg['train_split']}, "
              f"val: {val_split if use_native_val else '5% train carve'}, "
              f"test: {test_split or 'MISSING'})")
        print(f"[NER] Labels   : {list(ner_label_descriptions.keys())}")

        def _load_ner_examples(split_name, desc, cap=None):
            raw = _hf_load(ner_repo, split=split_name)
            if cap and len(raw) > cap:
                raw = raw.select(range(cap))
            allowed_ner = set(ner_label_descriptions.keys())
            records = [
                normalize_record_generic(x, allowed_labels=allowed_ner)
                for x in tqdm(raw, desc=f"normalize {desc}")
            ]
            return [
                ex
                for ex in (
                    to_input_example_generic(r, ner_label_descriptions)
                    for r in tqdm(records, desc=f"build {desc}")
                )
                if is_valid_example(ex)
            ]

        train_examples = _load_ner_examples(
            ner_cfg["train_split"], "ner train", cap=args.max_samples
        )

        if use_native_val:
            val_examples = _load_ner_examples(
                val_split, "ner val", cap=args.max_val_samples
            )
        else:
            # Carve out 5% as in-training validation (no data leakage from test set)
            _rng.seed(42)
            shuffled = train_examples[:]
            _rng.shuffle(shuffled)
            val_cut = max(1, int(len(shuffled) * 0.05))
            val_examples = shuffled[:val_cut]
            train_examples = shuffled[val_cut:]

        print(f"[NER] Train: {len(train_examples)}  Val: {len(val_examples)} "
              f"({'native split' if use_native_val else '5% train carve'})")

    elif args.fold is not None:
        if not (0 <= args.fold < N_FOLDS):
            raise ValueError(f"--fold must be in [0, {N_FOLDS - 1}], got {args.fold}")
        print(f"[K-Fold] Loading fold {args.fold} from {FOLD_DATASET_ID}")
        train_records, val_records, _ = load_fold_split(args.fold)
        train_examples = [
            ex for ex in (to_input_example(r, USE_LABEL_DESCRIPTIONS) for r in tqdm(train_records, desc="build train"))
            if is_valid_example(ex)
        ]
        val_examples = [
            ex for ex in (to_input_example(r, USE_LABEL_DESCRIPTIONS) for r in tqdm(val_records, desc="build val"))
            if is_valid_example(ex)
        ]
    else:
        max_train = args.max_samples
        lt_kwargs = {"max_train": max_train or 1_000_000}
        if args.dataset:
            lt_kwargs["dataset_id"] = args.dataset
        if args.max_val_samples is not None:
            lt_kwargs["max_val"] = args.max_val_samples
        train_records, val_records = load_train_val(**lt_kwargs)

        train_examples = [
            ex for ex in (to_input_example(r, USE_LABEL_DESCRIPTIONS) for r in tqdm(train_records, desc="build train"))
            if is_valid_example(ex)
        ]
        val_examples = [
            ex for ex in (to_input_example(r, USE_LABEL_DESCRIPTIONS) for r in tqdm(val_records, desc="build val"))
            if is_valid_example(ex)
        ]

    if args.augment:
        train_examples = augment_train_examples(train_examples)

    # ── Experiment-specific data pipeline ──────────────────────────────────
    config, private_opts = build_config(args)

    # CLI loss flags (orthogonal axis) override the experiment's loss settings.
    _merge_cli_loss(args, private_opts)

    if private_opts.get("_oversample"):
        print(f"[Exp] Oversampling rare labels (threshold={private_opts.get('_oversample_rare_threshold', 0.25)}, "
              f"max_factor={private_opts.get('_oversample_max_factor', 4.0)})")
        train_examples, os_stats = oversample_rare_labels(
            train_examples,
            rare_threshold=private_opts.get("_oversample_rare_threshold", 0.25),
            max_oversample_factor=private_opts.get("_oversample_max_factor", 4.0),
            seed=config.seed,
        )
        print_oversampling_stats(os_stats)

    if private_opts.get("_entity_aug"):
        freq = compute_label_frequencies(train_examples)
        rare_labels = detect_rare_labels(
            freq,
            rare_threshold=private_opts.get("_oversample_rare_threshold", 0.25),
        )
        print(f"[Exp] Entity-aware augmentation on {len(rare_labels)} rare labels")
        train_examples, ea_stats = entity_aware_augment(
            train_examples,
            rare_labels=rare_labels if rare_labels else None,
            aug_ratio=private_opts.get("_entity_aug_ratio", 0.4),
            aug_factor=private_opts.get("_entity_aug_factor", 1),
            seed=config.seed,
        )
        print_entity_aug_stats(ea_stats)

    # ── Value diversity pipeline (Exp 6 / 7) ──────────────────────────────
    if private_opts.get("_value_diversify"):
        div_thresh = private_opts.get("_value_diversify_threshold", 0.15)
        div_max = private_opts.get("_value_diversify_max_repeats", 5)
        low_div = detect_low_diversity_labels(train_examples, threshold=div_thresh)
        print(f"[Exp] Value diversification: {len(low_div)} low-diversity labels "
              f"(threshold={div_thresh}): {sorted(low_div)}")
        train_examples, div_stats = diversify_values(
            train_examples, target_labels=low_div,
            max_value_repeats=div_max, seed=config.seed,
        )
        print_diversity_stats(div_stats, strategy="replace")

    if private_opts.get("_value_mask"):
        mask_ratio = private_opts.get("_value_mask_ratio", 0.3)
        low_div = detect_low_diversity_labels(train_examples, threshold=0.1)
        print(f"[Exp] Context masking on {len(low_div)} low-diversity labels "
              f"(mask_ratio={mask_ratio})")
        train_examples, mask_stats = mask_entity_values(
            train_examples, target_labels=low_div,
            mask_ratio=mask_ratio, seed=config.seed,
        )
        print_diversity_stats(mask_stats, strategy="mask")

    if private_opts.get("_value_cap"):
        cap_max = private_opts.get("_value_cap_max_repeats", 8)
        low_div = detect_low_diversity_labels(train_examples, threshold=0.15)
        print(f"[Exp] Value occurrence cap (max={cap_max}) on "
              f"{len(low_div)} low-diversity labels")
        train_examples, cap_stats = cap_value_occurrences(
            train_examples, target_labels=low_div,
            max_value_repeats=cap_max, seed=config.seed,
        )
        print_diversity_stats(cap_stats, strategy="cap")

    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}")

    train_dataset = TrainingDataset(train_examples)
    val_dataset = TrainingDataset(val_examples)
    train_dataset.validate(raise_on_error=True)
    val_dataset.validate(raise_on_error=True)

    if args.warmstart_gliner2:
        from models.loader import build_warmstart_bi2enc_from_gliner2
        model = build_warmstart_bi2enc_from_gliner2(
            args.warmstart_gliner2, arch=args.arch, max_width=MAX_WIDTH,
        )
    else:
        proj_dim = "hidden" if str(args.proj_dim).lower() == "hidden" else int(args.proj_dim)
        model = load_gliner2_model(
            args.model, max_width=MAX_WIDTH, use_mmbert=args.mmbert,
            backbone=args.backbone, arch=args.arch, schema_projection_dim=proj_dim,
        )

    # Apply encoder freezing (Exp 4 / Exp 5 conservative options)
    if private_opts.get("_freeze_embeddings"):
        freeze_embeddings(model)
    if private_opts.get("_freeze_encoder_layers", 0):
        freeze_encoder_layers(model, private_opts["_freeze_encoder_layers"])
    if private_opts.get("_freeze_embeddings") or private_opts.get("_freeze_encoder_layers", 0):
        print_frozen_summary(model)

    # ── Orthogonal null anchor (works with any loss; --loss set implies it) ──
    if args.null_anchor and args.no_null_anchor:
        raise SystemExit("--null-anchor and --no-null-anchor are mutually exclusive.")
    if args.null_anchor:
        if hasattr(model, "configure_scorer"):
            _null_p = float(args.null_init) if args.null_init is not None else 0.7
            model.configure_scorer(null_score=True, null_init_p=_null_p)
            print(f"[null] learnable null anchor enabled (init p={_null_p}); "
                  "evaluate this checkpoint with --threshold 0.5")
        else:
            print("[null][warn] model has no configure_scorer; --null-anchor ignored")

    # ── Loss modifications (Exp 8 / 9 / 10 / 11 / 12 / 13 / 14) ────────────
    _bs_eps = float(private_opts.get("_loss_boundary_smoothing", 0) or 0)
    if _bs_eps > 0:
        # Boundary smoothing modifies labs_soft before the term sees it, so it
        # is compatible with focal_eiou (term operates on the soft target).
        # Still blocked for plain focal / eiou single-term until tested.
        if private_opts.get("_loss_focal") or private_opts.get("_loss_entity_iou"):
            raise SystemExit(
                "--boundary-smoothing is not supported with focal or eiou alone; use focal_eiou or bce."
            )
        if float(private_opts.get("_loss_label_smoothing", 0) or 0) > 0:
            raise SystemExit(
                "--boundary-smoothing and --loss-label-smoothing are mutually exclusive."
            )
    if private_opts.get("_loss_focal_eiou"):
        # Combined mix·focal + (1-mix)·eiou
        _fe_field_weights = None
        if private_opts.get("_loss_class_weight"):
            _fe_field_weights = compute_global_class_weights(
                train_examples,
                max_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                label_weight_overrides=private_opts.get("_loss_label_weight_overrides"),
            )
        apply_focal_eiou_loss(
            model,
            mix=float(private_opts.get("_loss_mix", 0.5)),
            gamma=float(private_opts.get("_loss_focal_gamma", 2.0)),
            focal_alpha=private_opts.get("_loss_focal_alpha", 0.25),
            eiou_alpha=float(private_opts.get("_loss_entity_iou_alpha", 0.25)),
            label_smoothing=float(private_opts.get("_loss_label_smoothing", 0)),
            boundary_smoothing=_bs_eps,
            boundary_smoothing_size=int(private_opts.get("_loss_boundary_smoothing_size", 1) or 1),
            field_weights=_fe_field_weights,
        )
    elif private_opts.get("_loss_entity_iou"):
        # Entity IoU Loss (Liu et al., 2022) — hard negative weighting via ENIoU
        _eiou_field_weights = None
        if private_opts.get("_loss_class_weight"):
            _eiou_field_weights = compute_global_class_weights(
                train_examples,
                max_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                label_weight_overrides=private_opts.get("_loss_label_weight_overrides"),
            )
        apply_entity_iou_loss(
            model,
            alpha=float(private_opts.get("_loss_entity_iou_alpha", 0.25)),
            label_smoothing=float(private_opts.get("_loss_label_smoothing", 0)),
            field_weights=_eiou_field_weights,
        )
    elif private_opts.get("_loss_focal"):
        # Focal loss replaces BCE entirely
        _fl_field_weights = None
        if private_opts.get("_loss_class_weight"):
            _fl_field_weights = compute_global_class_weights(
                train_examples,
                max_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                label_weight_overrides=private_opts.get("_loss_label_weight_overrides"),
            )
        apply_focal_loss(
            model,
            gamma=float(private_opts.get("_loss_focal_gamma", 2.0)),
            alpha=private_opts.get("_loss_focal_alpha", 0.25),
            label_smoothing=float(private_opts.get("_loss_label_smoothing", 0)),
            field_weights=_fl_field_weights,
        )
    elif private_opts.get("_loss_set"):
        # Set-Consistent: match the above-threshold pool {s > 0} vs gold
        # (lenient-IoU ownership) + learnable null anchor in the scorer
        # (decode at --threshold 0.5).
        if args.no_null_anchor:
            print("[set] null anchor SUPPRESSED (--no-null-anchor): the competition "
                  "terms act on raw logits, so logit 0 means sigmoid 0.5 again. "
                  "Evaluate this checkpoint at the legacy --threshold 0.7.")
        elif hasattr(model, "configure_scorer"):
            model.configure_scorer(
                null_score=True,
                null_init_p=float(private_opts.get("_loss_set_null_init", 0.7)),
            )
            print("[set] scorer null anchor enabled "
                  f"(init p={float(private_opts.get('_loss_set_null_init', 0.7))}); "
                  "evaluate this checkpoint with --threshold 0.5")
        _sc_field_weights = None
        if private_opts.get("_loss_class_weight"):
            _sc_field_weights = compute_global_class_weights(
                train_examples,
                max_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                label_weight_overrides=private_opts.get("_loss_label_weight_overrides"),
            )
        apply_set_consistent_loss(
            model,
            margin_set=float(private_opts.get("_loss_set_margin", 0.5)),
            margin_label=float(private_opts.get("_loss_set_margin_label", 0.5)),
            lambda_set=float(private_opts.get("_loss_set_lambda", 1.0)),
            lambda_fp=float(private_opts.get("_loss_set_lambda_fp", 0.25)),
            lambda_label=float(private_opts.get("_loss_set_lambda_label", 0.25)),
            set_topk=int(private_opts.get("_loss_set_topk", 0)),
            iou_match=float(private_opts.get("_loss_set_iou_match", 0.5)),
            base_margin=float(private_opts.get("_loss_set_base_margin", 0.0)),
            pos_weight=private_opts.get("_loss_pos_weight"),
            label_smoothing=float(private_opts.get("_loss_label_smoothing", 0)),
            boundary_smoothing=_bs_eps,
            boundary_smoothing_size=int(private_opts.get("_loss_boundary_smoothing_size", 1) or 1),
            field_weights=_sc_field_weights,
        )
    elif private_opts.get("_loss_margin"):
        # Additive-margin BCE (CosFace/ArcFace-style) replaces BCE entirely
        _mg_field_weights = None
        if private_opts.get("_loss_class_weight"):
            _mg_field_weights = compute_global_class_weights(
                train_examples,
                max_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                label_weight_overrides=private_opts.get("_loss_label_weight_overrides"),
            )
        apply_margin_loss(
            model,
            margin=float(private_opts.get("_loss_margin_value", 0.15)),
            label_smoothing=float(private_opts.get("_loss_label_smoothing", 0)),
            boundary_smoothing=_bs_eps,
            boundary_smoothing_size=int(private_opts.get("_loss_boundary_smoothing_size", 1) or 1),
            field_weights=_mg_field_weights,
        )
    else:
        _has_loss_mod = (
            private_opts.get("_loss_pos_weight") is not None
            or private_opts.get("_loss_class_weight")
            or private_opts.get("_loss_label_smoothing", 0) > 0
            or _bs_eps > 0
            or args.full_grid
        )
        if _has_loss_mod:
            # Compute global per-label weights from training data
            _field_weights = None
            if private_opts.get("_loss_class_weight"):
                _label_overrides = private_opts.get("_loss_label_weight_overrides")
                _field_weights = compute_global_class_weights(
                    train_examples,
                    max_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                    label_weight_overrides=_label_overrides,
                )
                # Print top weighted labels for visibility
                if _field_weights:
                    sorted_w = sorted(_field_weights.items(), key=lambda x: x[1], reverse=True)
                    print(f"[LossWrapper] Global field weights ({len(sorted_w)} labels):")
                    for lbl, w in sorted_w[:10]:
                        print(f"  {lbl}: {w:.3f}")
                    if len(sorted_w) > 10:
                        print(f"  ... ({len(sorted_w) - 10} more)")

            apply_weighted_loss(
                model,
                pos_weight=private_opts.get("_loss_pos_weight"),
                field_weights=_field_weights,
                class_weight=bool(private_opts.get("_loss_class_weight")) and _field_weights is None,
                max_class_weight=float(private_opts.get("_loss_max_class_weight", 10.0)),
                label_smoothing=float(private_opts.get("_loss_label_smoothing", 0)),
                boundary_smoothing=_bs_eps,
                boundary_smoothing_size=int(private_opts.get("_loss_boundary_smoothing_size", 1) or 1),
                full_grid=args.full_grid,
            )
    # feat/affine_adapt: swap the span scorer (dot -> biaffine/cosine) and set the
    # freeze tier per the experiment file. No-op when --affine-exp is omitted.
    affine_exp = None
    if args.affine_exp:
        from experiments.affine.loader import apply_experiment, load_experiment
        affine_exp = load_experiment(args.affine_exp)
        apply_experiment(model, affine_exp)

    config, _ = build_config(args, overrides=(affine_exp or {}).get("train_overrides"))
    if affine_exp:
        config.experiment_name = f"{config.experiment_name}-{affine_exp['name']}"

    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=train_dataset, eval_data=val_dataset)

    repo = args.repo
    if args.no_push:
        print("[no-push] Skipping HF Hub upload (--no-push). "
              f"Checkpoint saved locally under {args.output_dir}")
        return
    affine_tag = f"-{affine_exp['name']}" if affine_exp else ""
    if repo is None and args.fold is not None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        repo = f"{HF_ORG}/gliner2-fold{args.fold}{affine_tag}-{date_str}"
        print(f"[K-Fold] Auto repo: {repo}")
    elif repo is None and args.warmstart_gliner2:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        repo = f"{HF_ORG}/gliner2-warmstart-{_ARCH_TAG.get(args.arch, args.arch)}{affine_tag}-{date_str}"
        print(f"[Warmstart] Auto repo: {repo}")
    elif repo is None and args.backbone is not None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        repo = f"{HF_ORG}/gliner2-{args.backbone}-{args.arch}{_proj_tag(args)}{affine_tag}-{date_str}"
        print(f"[Backbone] Auto repo: {repo}")
    elif repo is None and args.experiment and args.experiment != "baseline":
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        repo = f"{HF_ORG}/gliner2-{args.experiment}-{date_str}"
        print(f"[Experiment] Auto repo: {repo}")
    if repo:
        # Ensure repo is under the org namespace
        if "/" not in repo:
            repo = f"{HF_ORG}/{repo}"
        best_dir = os.path.join(args.output_dir, "best")
        if not os.path.exists(best_dir):
            best_dir = os.path.join(args.output_dir, "final")
        _push_to_hub(best_dir, repo)


if __name__ == "__main__":
    main()
