"""GLiNER and GLiNER2 BCE baselines on four PII corpora, three seeds each, on Modal.

What this runs
--------------
4 datasets (privy, kiji, gretel, pii_500k) x 2 model families x 3 seeds = 24 runs.

    gliner2   training/train.py with the stock BCE objective (no --loss flag) on
              ``fastino/gliner2-multi-v1``, then evaluation/evaluate.py.
    gliner    scripts/run_gliner_large_multiseed.py --loss bce on GLiNER v1.  The
              checkpoint is per-dataset (``GLINER_CHECKPOINTS``): gliner_large-v2 for the
              English-only corpora, gliner_multi-v2.1 for the multilingual ones, because
              deberta-v3-large cannot read the six/eight languages in kiji and pii_500k
              and an English-only backbone there would confound the family comparison.

Both families dump ``<split>/predictions/all.jsonl`` in evaluate.py's schema into the
results Volume, so ``scripts/score_pii_baselines.py`` can score the two families with one
strict scorer (HungarianEvaluator, IoU 1.0, no fuzzy) rather than two look-alike ones.

Usage (one Modal profile per dataset keeps each workspace's quota independent):

    modal run scripts/modal_pii_baselines.py --action prepare          # build + push data
    modal run scripts/modal_pii_baselines.py --action smoke --datasets privy
    modal run scripts/modal_pii_baselines.py --action run --datasets privy
    modal volume get pii-baselines-results / ./out_pii_baselines
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

import modal

APP_NAME = "pii-baselines"
RESULT_VOLUME_NAME = "pii-baselines-results"
CACHE_VOLUME_NAME = "pii-hf-cache"
REMOTE_REPO = "/workspace"
REMOTE_RESULTS = "/results"

PROTOCOL = "v1-bce-baseline-50k"

# The objective axis. Everything else -- data, epochs, batch, GPU, seeds, eval -- is held
# identical to the baseline, which is the only thing that makes the comparison mean
# anything. `set` is the paper's C_cw_pw2 recipe (NER_cw_pw2_10seed.ipynb): set-consistent
# term with the uncapped competition pool, inverse-frequency class weights, pos_weight 2 and
# boundary smoothing. On GLiNER2 `--loss set` also switches the scorer's null anchor on, so
# the operating point is pinned at sigmoid 0.5 -- hence THRESHOLD 0.5 for every run here.
RECIPES = {
    "bce": {"protocol": "v1-bce-baseline-50k", "gliner2": [], "gliner": []},
    "set": {
        "protocol": "v1-set-consistent-50k",
        "gliner2": ["--loss", "set", "--set-topk", "0", "--loss-class-weight",
                    "--pos-weight", "2", "--boundary-smoothing", "0.1",
                    "--boundary-smoothing-size", "1"],
        "gliner": ["--loss", "set", "--set-topk", "0", "--class-weight",
                   "--pos-weight", "2", "--boundary-smoothing", "0.1",
                   "--boundary-smoothing-size", "1"],
    },
    # PROTOCOL.md arm (b): everything `set` turns on except the three competition terms --
    # class weights, pos_weight 2, boundary smoothing, the null anchor and the full grid
    # (no negative masking); `--loss set` enables the last two implicitly, so they are
    # named here. (c) minus (b) isolates the terms.
    "bce_bundle": {
        "protocol": "v1-bce-bundle-50k",
        "gliner2": ["--loss", "bce", "--loss-class-weight", "--pos-weight", "2",
                    "--boundary-smoothing", "0.1", "--boundary-smoothing-size", "1",
                    "--null-anchor", "--full-grid"],
        "gliner": None,   # no null-anchor scorer in GLiNER v1
    },
    # No training: fastino/gliner2-multi-v1 decodes each corpus with that corpus's own schema
    # and descriptions. The consequence test of the audit (PROTOCOL.md 9.7) on corpora where
    # no model was trained, and a memorisation-free control for the seen-vs-novel-value gap.
    "zeroshot": {"protocol": "v1-zeroshot", "gliner2": [], "gliner": []},
    # PROTOCOL.md 9.9 arm (a'): the stock objective with negative masking switched off and
    # nothing else of (b) -- how much of (b)-(a) the full grid alone accounts for.
    "bce_fullgrid": {"protocol": "v1-bce-fullgrid-50k", "gliner2": ["--loss", "bce", "--full-grid"],
                     "gliner": None},
}
DATASETS = ("privy", "kiji", "gretel", "pii_500k")
SEEDS = (42, 1337, 2024)
GLINER2_BASE = "fastino/gliner2-multi-v1"

# English-only corpora keep the paper's GLiNER-L checkpoint; the multilingual ones get the
# multilingual GLiNER so the v1 baseline is not handicapped by its backbone's language.
GLINER_CHECKPOINTS = {
    "privy": "urchade/gliner_large-v2",
    "gretel": "urchade/gliner_large-v2",
    "kiji": "urchade/gliner_multi-v2.1",
    "pii_500k": "urchade/gliner_multi-v2.1",
}

EPOCHS = 3
THRESHOLD = 0.5

# Effective batch is 16 for both families; the split differs because GLiNER v1's span
# head is far heavier. On A10G at batch 8 its trainer logged "Skipping batch due to CUDA
# OOM" -- it drops those batches silently, which would quietly shrink the training set,
# so the v1 family gets an A100 and a smaller micro-batch instead.
BATCH = {"gliner2": (8, 2), "gliner": (4, 4)}
GPU = {"gliner2": "A10G", "gliner": "A100-40GB"}
# privy's texts run to 4.7k chars (XML/JSON blobs), and GLiNER2 on A10G logged
# "OOM at step N, batch skipped" on ~all of the first 200 steps -- its trainer drops the
# batch too. More memory rather than a smaller batch, so every dataset in the table keeps
# the identical effective batch of 16.
# privy's tail (up to 4.7k chars of XML/SQL) still skipped ~2% of batches on a 40 GiB
# A100 even at micro-batch 4 -- and the batches it drops are exactly the longest
# documents, so the loss is systematic, not random. It gets the 80 GiB card.
GPU_OVERRIDE = {
    ("gliner2", "privy"): "A100-80GB",
    ("gliner", "privy"): "A100-80GB",
    ("gliner2", "gretel"): "A100-40GB",   # 42 labels x multi-line documents; 0 OOM measured
    # ai4p_300k skipped 94-135 batches per run on A10G at 8x2 (PROTOCOL.md 9.8)
    ("gliner2", "ai4p_300k"): "A100-40GB",
    # long documents and 29/55 types, like gretel (PROTOCOL.md 9.11 addendum)
    ("gliner2", "nemotron_pii"): "A100-40GB",
    ("gliner2", "gretel_finance"): "A100-40GB",
    # TAB: 1,014 court judgments, chunks up to ~1.3k splitter tokens (PROTOCOL.md 9.15)
    ("gliner2", "tab"): "H200",
}
# privy still skipped ~11 batches on a 40 GiB A100 at micro-batch 8, so the two heavy
# corpora halve the micro-batch and double accumulation: peak memory drops, the effective
# batch every run optimises against stays 16.
BATCH_OVERRIDE = {
    ("gliner2", "privy"): (4, 4),
    ("gliner2", "gretel"): (4, 4),
    ("gliner2", "ai4p_300k"): (4, 4),
    ("gliner2", "nemotron_pii"): (4, 4),
    ("gliner2", "gretel_finance"): (4, 4),
    ("gliner2", "tab"): (4, 4),
}
# GLiNER2 defaults to fp16 (warm start, which the project notes record as safe for this
# deberta-v2 encoder). On privy it is not: seed 42 trained cleanly to epoch 2.2 and then
# went loss=NaN for the rest of the run -- and gliner2/training/trainer.py has no NaN
# guard, so it kept calling optimizer.step(), wrote a poisoned checkpoint and evaluate.py
# died on it. Seed 1337 on identical settings finished fine (macro 99.19), so this is
# data-dependent fp16 overflow, not a broken config. bf16 keeps the speed and the range.
DTYPE_OVERRIDE = {("gliner2", "privy"): "bf16"}


def dtype_for(family: str, dataset: str) -> str:
    return DTYPE_OVERRIDE.get((family, dataset), "fp16" if family == "gliner2" else "bf16")


def gpu_for(family: str, dataset: str) -> str:
    return GPU_OVERRIDE.get((family, dataset), GPU[family])


def batch_for(family: str, dataset: str) -> tuple:
    return BATCH_OVERRIDE.get((family, dataset), BATCH[family])


def _image() -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "torch",
            "gliner2",
            "gliner",
            "datasets",
            "pandas",
            "tqdm",
            "huggingface_hub",
            "hf_xet",
            "scikit-learn",
            "chonkie",
            "transformers>=4.48,<5",
            "scipy",
            "accelerate",
        )
    )
    for directory in ("components", "config", "data", "evaluation", "gliner2",
                      "models", "training", "scripts"):
        image = image.add_local_dir(directory, f"{REMOTE_REPO}/{directory}", copy=True)
    return image


image = _image()
app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _hf_secret() -> List[modal.Secret]:
    """The HF write token, needed only by `prepare`. Absent secret -> no push."""
    try:
        return [modal.Secret.from_name("huggingface")]
    except Exception:
        return []


def _run(command: List[str], *, cwd: str = REMOTE_REPO) -> None:
    print("$", " ".join(command), flush=True)
    env = {**os.environ,
           "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(command, cwd=cwd, check=True, env=env)


# A run that skips a batch or goes NaN is unusable (PROTOCOL.md 9.8), and the
# trainer does neither loudly: it logs and carries on for hours. Stop at the first sign.
_FATAL_TRAIN_LINES = ("batch skipped", "loss=nan", "loss=NaN")


def _run_train(command: List[str], *, cwd: str = REMOTE_REPO, abort: bool = True) -> Dict[str, int]:
    """Stream training output. abort=True stops at the first fatal line; abort=False only
    counts them (PROTOCOL.md 9.14: measure what the original, unguarded runs skipped)."""
    print("$", " ".join(command), flush=True)
    env = {**os.environ,
           "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    counts = {"skipped_batches": 0, "nan_lines": 0}
    for line in proc.stdout:
        print(line, end="", flush=True)
        hit = next((f for f in _FATAL_TRAIN_LINES if f in line), None)
        if hit:
            counts["skipped_batches" if hit == "batch skipped" else "nan_lines"] += 1
            if abort:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"training aborted at first {hit!r}: {line.strip()[:200]}")
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return counts


def run_dir(family: str, dataset: str, seed: int, tag: str = PROTOCOL) -> pathlib.Path:
    """Results live under a tag so a smoke run can never be mistaken for a real one."""
    return pathlib.Path(REMOTE_RESULTS) / tag / family / dataset / f"seed-{seed}"


# -----------------------------------------------------------------------------
# Data preparation (CPU)
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 3,
    secrets=_hf_secret(),
    volumes={REMOTE_RESULTS: results_volume, "/root/.cache/huggingface": cache_volume},
)
def prepare(datasets: List[str], max_train: int, max_val: int, max_test: int,
            hf_user: str, max_train_chars: int = 2000) -> str:
    """Convert the four public corpora and push them to the Hub (see prepare_pii_datasets)."""
    command = [
        sys.executable, "scripts/prepare_pii_datasets.py",
        "--datasets", *datasets,
        "--max-train", str(max_train),
        "--max-val", str(max_val),
        "--max-test", str(max_test),
        "--max-train-chars", str(max_train_chars),
        "--hf-user", hf_user,
        "--push",
        "--private",
        "--stats-out", f"{REMOTE_RESULTS}/{PROTOCOL}/pii_dataset_stats.json",
    ]
    pathlib.Path(f"{REMOTE_RESULTS}/{PROTOCOL}").mkdir(parents=True, exist_ok=True)
    _run(command)
    results_volume.commit()
    return pathlib.Path(f"{REMOTE_RESULTS}/{PROTOCOL}/pii_dataset_stats.json").read_text()


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 30,
    secrets=_hf_secret(),
    volumes={REMOTE_RESULTS: results_volume},
)
def push_checkpoint(rel_dir: str, repo_id: str) -> str:
    from huggingface_hub import HfApi
    folder = pathlib.Path(REMOTE_RESULTS) / rel_dir
    if not (folder / "config.json").is_file():
        raise FileNotFoundError(f"no checkpoint at {folder}")
    api = HfApi()   # token from HF_TOKEN or HUGGING_FACE_HUB_TOKEN, whichever the secret sets
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id)
    return repo_id


# -----------------------------------------------------------------------------
# One training + evaluation run
# -----------------------------------------------------------------------------

def _gpu_function(gpu: str):
    """Same body, one Modal Function per GPU type (gpu= is fixed at decoration time)."""
    return app.function(
        image=image,
        gpu=gpu,
        cpu=4,
        memory=32768,
        timeout=60 * 60 * 8,
        max_containers=6,
        retries=modal.Retries(max_retries=1, backoff_coefficient=2.0, initial_delay=30.0),
        secrets=_hf_secret(),
        volumes={REMOTE_RESULTS: results_volume, "/root/.cache/huggingface": cache_volume},
        name=f"run_job_{gpu.replace('-', '_').lower()}",
    )


def _local_model_copy(model_id: str, work: pathlib.Path):
    """Download a released GLiNER2 checkpoint and make its tokenizer loadable here.

    fastino/gliner2-privacy-filter-PII-multi was saved by transformers 5, whose
    tokenizer_config.json lists `extra_special_tokens`; transformers 4.x expects a dict
    there and fails. The same tokens are in tokenizer.json's added tokens, so dropping the
    key leaves tokenization unchanged (checked locally: [SEP_TEXT], [P], [E], ... stay single
    tokens, vocabulary 250,112). Returns (local path, whether the file was patched).
    """
    from huggingface_hub import snapshot_download
    local = work / ("model-" + model_id.replace("/", "__"))
    shutil.copytree(snapshot_download(model_id), local, dirs_exist_ok=True)
    cfg_path = local / "tokenizer_config.json"
    patched = False
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg.pop("extra_special_tokens")
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            patched = True
    return str(local), patched


def _execute(job: Dict[str, Any]) -> Dict[str, Any]:
    """Train one (family, dataset, seed) and leave prediction dumps in the Volume."""
    started = time.time()
    family, dataset, seed = job["family"], job["dataset"], job["seed"]
    recipe = job.get("recipe", "bce")
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; known: {list(RECIPES)}")
    if RECIPES[recipe].get(family) is None:
        raise ValueError(f"recipe {recipe!r} has no {family} variant")
    # job["gpu"] pins every run to one card (e.g. RTX-PRO-6000) instead of the per-dataset
    # table; the caller must spawn it on the matching Function (launch_pii_jobs.py does).
    gpu = job.get("gpu") or gpu_for(family, dataset)
    # One smoke dir per recipe (and per pinned GPU), so smoke runs never overwrite each other.
    # job["round"] (e.g. "v2") keeps a new round's runs out of the pilot's v1-* directories,
    # where an existing status=ok record would otherwise make them skip.
    if job.get("smoke"):
        tag = f"smoke-{recipe}" + (f"-{job['gpu']}" if job.get("gpu") else "")
    else:
        tag = RECIPES[recipe]["protocol"].replace("v1-", f"{job.get('round', 'v1')}-", 1)
    if job.get("model"):
        tag += "-" + job["model"].split("/")[-1].lower()
    # PROTOCOL.md 9.18: label-name intervention gets its own directory
    if job.get("label_names", "original") != "original":
        tag += {"description": "-labelpara", "humanized": "-labelhuman"}[job["label_names"]]
    # dump_threshold: a second decode of each split at a low threshold, kept next to the
    # official one, so threshold analyses (PROTOCOL.md) run offline without the checkpoint.
    dump_threshold = job.get("dump_threshold")
    out = run_dir(family, dataset, seed, tag)
    record_path = out / "record.json"

    results_volume.reload()
    if record_path.exists() and not job.get("force"):
        record = json.loads(record_path.read_text())
        if record.get("status") == "ok":
            print(f"[skip] {family}/{dataset}/seed-{seed} already complete")
            return record

    work = pathlib.Path(f"/tmp/{family}-{dataset}-{seed}")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    batch_size, grad_accum = batch_for(family, dataset)
    record: Dict[str, Any] = {
        **job,
        "protocol": tag,
        "recipe": recipe,
        "gpu": gpu,
        "dtype": dtype_for(family, dataset),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "epochs": EPOCHS,
        "threshold": THRESHOLD,
        "train_flags": RECIPES[recipe][family],
        "status": "running",
    }

    # evaluate.py does not read the repo out of the registry -- it must be told.
    sys.path.insert(0, REMOTE_REPO)
    from config.ner_datasets import NER_DATASETS
    repo = NER_DATASETS[dataset]["repo"]
    record["repo"] = repo
    cap_train = job.get("max_train_samples")
    cap_eval = job.get("max_eval_samples")
    # a test-only release (MAPA) declares no validation split in the registry;
    # job["splits"] narrows further (the zero-shot rounds decode test only)
    splits = ("validation", "test") if NER_DATASETS[dataset].get("val_splits", ["validation"]) else ("test",)
    if job.get("splits"):
        splits = tuple(s for s in splits if s in job["splits"])

    try:
        if family == "gliner2":
            record["base_model"] = GLINER2_BASE
            ckpt = work / "checkpoint"
            train_command = [
                sys.executable, "training/train.py",
                "--model", GLINER2_BASE,
                # job["train_dataset"]: train on another registry key (TAB trains on its
                # chunked copy) while evaluation and the run directory stay on `dataset`
                "--ner-dataset", job.get("train_dataset") or dataset,
                "--epochs", str(EPOCHS),
                "--batch-size", str(batch_size),
                "--grad-accum", str(grad_accum),
                "--dtype", dtype_for(family, dataset),
                "--seed", str(seed),
                "--output-dir", str(ckpt),
                "--no-push",
                *RECIPES[recipe]["gliner2"],
            ]
            if cap_train:
                train_command += ["--max-samples", str(cap_train),
                                  "--max-val-samples", str(cap_eval or 32)]
            if recipe == "zeroshot":          # evaluate a released model as is
                model_path = job.get("model") or GLINER2_BASE
                record["base_model"] = model_path
                if job.get("model"):
                    model_path, record["tokenizer_patch"] = _local_model_copy(job["model"], work)
            else:
                record["train_guard"] = _run_train(train_command, abort=not job.get("count_skips"))
                best = ckpt / "best"
                model_path = str(best if best.is_dir() else ckpt)
            decodes = [(THRESHOLD, "")]
            if dump_threshold is not None:
                decodes.append((float(dump_threshold), f"_t{dump_threshold}"))
            for split in splits:
                for threshold, suffix in decodes:
                    eval_command = [
                        sys.executable, "evaluation/evaluate.py",
                        "--model", model_path,
                        "--ner-dataset", dataset,
                        "--dataset", repo,
                        "--split", split,
                        "--use-desc",
                        "--schema-case", "original",
                        "--no-extra-val",
                        "--threshold", str(threshold),
                        "--output-dir", str(out / f"{split}{suffix}"),
                    ]
                    if cap_eval:
                        eval_command += ["--max-samples", str(cap_eval)]
                    if job.get("label_names", "original") != "original":
                        eval_command += ["--label-names", job["label_names"]]
                    if job.get("infer_batch_size") is not None:   # long documents: batch 8 OOMs
                        eval_command += ["--infer-batch-size", str(job["infer_batch_size"])]
                    _run(eval_command)
                    # evaluate.py records a per-sample exception (e.g. CUDA OOM in batched
                    # inference) as an empty prediction and still exits 0 -- the zero-shot
                    # gretel_finance run lost 4,900 of 5,000 documents that way. Fail loudly.
                    dumped = out / f"{split}{suffix}" / "predictions" / "all.jsonl"
                    failed = sum(1 for line in dumped.open(encoding="utf-8")
                                 if line.strip() and json.loads(line).get("error"))
                    if failed:
                        raise RuntimeError(f"{failed} {split} samples failed inside evaluate.py")
            if job.get("keep_ckpt"):
                shutil.copytree(model_path, out / "checkpoint", dirs_exist_ok=True)
                record["checkpoint"] = str(out / "checkpoint")
        elif family == "gliner" and recipe == "zeroshot":
            # off-the-shelf GLiNER v1 PII checkpoint, no training (PROTOCOL.md 9.10)
            record["base_model"] = job["model"]
            for split in splits:
                command = [sys.executable, "scripts/eval_gliner_zeroshot.py",
                           "--model", job["model"], "--ner-dataset", dataset, "--split", split,
                           "--threshold", str(THRESHOLD), "--output-dir", str(out / split)]
                if cap_eval:
                    command += ["--max-samples", str(cap_eval)]
                _run(command)
                dumped = out / split / "predictions" / "all.jsonl"
                failed = sum(1 for line in dumped.open(encoding="utf-8")
                             if line.strip() and json.loads(line).get("error"))
                if failed:
                    raise RuntimeError(f"{failed} {split} documents failed in eval_gliner_zeroshot.py")
        elif family == "gliner":
            checkpoint = GLINER_CHECKPOINTS[dataset]
            record["base_model"] = checkpoint
            gliner_command = [
                sys.executable, "scripts/run_gliner_large_multiseed.py",
                "--model", checkpoint,
                "--datasets", dataset,
                "--seeds", str(seed),
                "--loss", "bce",
                "--epochs", str(EPOCHS),
                "--batch-size", str(batch_size),
                "--eval-batch-size", str(batch_size),
                "--gradient-accumulation-steps", str(grad_accum),
                "--dtype", dtype_for(family, dataset),
                "--threshold", str(THRESHOLD),
                "--eval-splits", "validation", "test",
                "--output-dir", str(work / "ckpt"),
                "--eval-dir", str(work / "eval"),
                "--cleanup",
                *RECIPES[recipe]["gliner"],
            ]
            if cap_train:
                gliner_command += ["--max-train-samples", str(cap_train),
                                   "--max-val-samples", str(cap_eval or 32),
                                   "--max-test-samples", str(cap_eval or 32)]
            _run(gliner_command)
            # The runner lays results out as <eval-dir>/<ds>__s<seed>/<split>/...
            produced = work / "eval" / f"{dataset}__s{seed}"
            for split_dir in sorted(produced.iterdir()):
                if split_dir.is_dir():
                    shutil.copytree(split_dir, out / split_dir.name, dirs_exist_ok=True)
            runner_results = work / "ckpt" / f"{dataset}__s{seed}" / "results.json"
            if runner_results.is_file():
                record["runner_metrics"] = json.loads(runner_results.read_text())["metrics_by_split"]
        else:
            raise ValueError(f"unknown family {family!r}")

        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        # A guard abort (skipped batch / NaN) is deterministic for this config: returning
        # instead of raising keeps Modal's retry from re-running the same doomed hours.
        if not str(exc).startswith("training aborted"):
            raise
    finally:
        record["elapsed_seconds"] = round(time.time() - started, 1)
        out.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        results_volume.commit()
        shutil.rmtree(work, ignore_errors=True)
    return record


run_job_a10g = _gpu_function("A10G")(_execute)
run_job_a100 = _gpu_function("A100-40GB")(_execute)
run_job_a100_80 = _gpu_function("A100-80GB")(_execute)
run_job_rtx_pro_6000 = _gpu_function("RTX-PRO-6000")(_execute)   # 96 GB, Blackwell
run_job_h200 = _gpu_function("H200")(_execute)                   # 141 GB
RUNNER = {"A10G": run_job_a10g, "A100-40GB": run_job_a100, "A100-80GB": run_job_a100_80,
          "RTX-PRO-6000": run_job_rtx_pro_6000, "H200": run_job_h200}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    action: str = "run",
    datasets: str = ",".join(DATASETS),
    families: str = "gliner2,gliner",
    seeds: str = ",".join(str(s) for s in SEEDS),
    hf_user: str = "anonymous-org",
    max_train: int = 50_000,
    max_val: int = 2_500,
    max_test: int = 5_000,
    max_train_chars: int = 2_000,
    force: bool = False,
) -> None:
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}; known: {list(DATASETS)}")

    if action == "prepare":
        print(prepare.remote(selected, max_train, max_val, max_test, hf_user,
                             max_train_chars))
        return

    family_list = [f.strip() for f in families.split(",") if f.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    if action == "smoke":
        # One tiny run per family to prove the container, data and CLI wiring.
        jobs = [
            {"family": family, "dataset": selected[0], "seed": seed_list[0],
             "smoke": True, "force": True,
             "max_train_samples": 64, "max_eval_samples": 32}
            for family in family_list
        ]
    elif action == "run":
        jobs = [
            {"family": family, "dataset": dataset, "seed": seed, "force": force}
            for dataset in selected
            for family in family_list
            for seed in seed_list
        ]
    else:
        raise SystemExit("--action must be prepare, smoke or run")

    print(f"Submitting {len(jobs)} job(s): {sorted({j['dataset'] for j in jobs})} "
          f"x {family_list} x {seed_list}")
    # Spawn rather than map: the two families are separate Modal Functions (different GPU
    # types), and spawning lets them run concurrently instead of one family after the other.
    handles = [(job, RUNNER[gpu_for(job["family"], job["dataset"])].spawn(job))
               for job in jobs]
    records: List[Dict[str, Any]] = []
    for job, handle in handles:
        try:
            records.append(handle.get())
        except Exception as exc:  # one failed run must not lose the other 23
            records.append({**job, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in records if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(jobs)} job(s) finished ok")
    for record in records:
        print(f"  {record['family']:<8s} {record['dataset']:<10s} "
              f"seed={record['seed']:<5d} {record['status']:<6s} "
              f"({record.get('elapsed_seconds', 0) / 60:.1f} min)"
              + (f"  {record.get('error', '')}" if record["status"] != "ok" else ""))
