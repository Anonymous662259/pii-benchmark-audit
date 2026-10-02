#!/usr/bin/env python3
"""Deploy the PII-baseline app and spawn runs that outlive this process.

Why this exists: ``modal run`` creates an *ephemeral* app tied to the client process.
When the local session ended, all four workspaces' apps were torn down mid-training and
24 runs were lost with nothing written.  A *deployed* app's ``.spawn()`` calls are
server-side: they keep running after this script exits, and they write their results into
the Volume, which ``scripts/pull_pii_results.py`` syncs down.

    python scripts/launch_pii_jobs.py --profile <workspace> --deploy \
        --datasets privy --families gliner2 --seeds 42,1337,2024
    python scripts/launch_pii_jobs.py --profile <workspace> --status

Spawned call ids are appended to ``out_pii_baselines/spawned_calls.json`` so a later
``--status`` can ask Modal what happened to each one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_NAME = "pii-baselines"
CALLS_FILE = Path("out_pii_baselines/spawned_calls.json")


def _function_name(gpu: str) -> str:
    return f"run_job_{gpu.replace('-', '_').lower()}"


def deploy(profile: str) -> None:
    env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8"}
    print(f"[{profile}] modal deploy ...", flush=True)
    proc = subprocess.run(
        ["modal", "deploy", "scripts/modal_pii_baselines.py"],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise SystemExit(f"deploy failed for {profile}:\n{proc.stdout}\n{proc.stderr}")
    print(f"[{profile}] deployed", flush=True)


def load_calls() -> List[Dict[str, Any]]:
    if CALLS_FILE.is_file():
        try:
            return json.loads(CALLS_FILE.read_text(encoding="utf-8"))
        except ValueError:
            return []
    return []


def save_calls(calls: List[Dict[str, Any]]) -> None:
    CALLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALLS_FILE.write_text(json.dumps(calls, indent=2), encoding="utf-8")


def spawn(profile: str, jobs: List[Dict[str, Any]]) -> None:
    os.environ["MODAL_PROFILE"] = profile
    import modal
    from scripts.modal_pii_baselines import gpu_for

    calls = load_calls()
    for job in jobs:
        gpu = job.get("gpu") or gpu_for(job["family"], job["dataset"])
        fn = modal.Function.from_name(APP_NAME, _function_name(gpu))
        call = fn.spawn(job)
        calls.append({"profile": profile, "call_id": call.object_id, "gpu": gpu, **job})
        print(f"  spawned {job.get('recipe','bce'):4s} {job['family']:8s} "
              f"{job['dataset']:9s} seed={job['seed']:<5d} {gpu:10s} -> {call.object_id}",
              flush=True)
    save_calls(calls)


def status(profile: str) -> None:
    os.environ["MODAL_PROFILE"] = profile
    import modal

    for entry in load_calls():
        if entry["profile"] != profile:
            continue
        call = modal.FunctionCall.from_id(entry["call_id"])
        try:
            result = call.get(timeout=0)
            state = f"done status={result.get('status')}"
        except TimeoutError:
            state = "running"
        except Exception as exc:
            state = f"{type(exc).__name__}: {str(exc)[:80]}"
        print(f"  {entry['family']:8s} {entry['dataset']:9s} seed={entry['seed']:<5d} "
              f"{entry['call_id']}  {state}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--deploy", action="store_true", help="Deploy before spawning.")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--families", default="gliner2,gliner")
    parser.add_argument("--seeds", default="42,1337,2024")
    parser.add_argument("--recipe", default="bce", choices=["bce", "set", "bce_bundle", "zeroshot", "bce_fullgrid"],
                        help="Objective: 'bce' baseline, 'set' (paper's C_cw_pw2), or "
                             "'bce_bundle' (set's extras without the competition terms).")
    parser.add_argument("--dump-threshold", type=float, default=None,
                        help="Also decode every split at this low threshold (e.g. 0.1) into "
                             "<split>_t<thr>/, for offline threshold analysis.")
    parser.add_argument("--keep-ckpt-seeds", default="",
                        help="Comma-separated seeds whose checkpoint is copied to the Volume.")
    parser.add_argument("--gpu", default=None,
                        choices=["A10G", "A100-40GB", "A100-80GB", "RTX-PRO-6000", "H200"],
                        help="Pin every run to this GPU instead of the per-dataset table in "
                             "modal_pii_baselines.py (batch/dtype overrides still apply).")
    parser.add_argument("--round", default="v1",
                        help="Result-directory prefix: v1-<recipe>-50k is the pilot; use v2 "
                             "for the PROTOCOL.md round so its seeds do not collide with it.")
    parser.add_argument("--splits", default="",
                        help="Comma list of splits to decode (default: validation,test).")
    parser.add_argument("--count-skips", action="store_true",
                        help="Do not abort on skipped batches / NaN; count them into the record (diagnostic).")
    parser.add_argument("--model", default=None,
                        help="zeroshot recipe: the released checkpoint to evaluate (default GLiNER2 base).")
    parser.add_argument("--infer-batch-size", type=int, default=None,
                        help="Passed to evaluate.py; 1 for corpora whose long documents OOM at 8.")
    parser.add_argument("--label-names", default="original", choices=["original", "description", "humanized"],
                        help="PROTOCOL.md 9.18: show each type's description as its name at eval.")
    parser.add_argument("--train-dataset", default=None,
                        help="Train on this registry key instead (e.g. tab_chunked); evaluation "
                             "and the result directory stay on --datasets.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if the Volume already holds an ok record.")
    parser.add_argument("--status", action="store_true", help="Report on spawned calls only.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny caps + results under the 'smoke' tag; proves the wiring.")
    args = parser.parse_args()

    if args.deploy:
        deploy(args.profile)

    if args.status:
        status(args.profile)
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise SystemExit("--datasets is required when spawning")
    extra = ({"smoke": True, "max_train_samples": 64, "max_eval_samples": 32}
             if args.smoke else {})
    if args.dump_threshold is not None:
        extra["dump_threshold"] = args.dump_threshold
    if args.gpu:
        extra["gpu"] = args.gpu
    if args.round != "v1":
        extra["round"] = args.round
    if args.model:
        extra["model"] = args.model
    if args.count_skips:
        extra["count_skips"] = True
    if args.infer_batch_size is not None:
        extra["infer_batch_size"] = args.infer_batch_size
    if args.train_dataset:
        extra["train_dataset"] = args.train_dataset
    if args.label_names != "original":
        extra["label_names"] = args.label_names
    if args.splits:
        extra["splits"] = [x.strip() for x in args.splits.split(",") if x.strip()]
    keep = {int(s) for s in args.keep_ckpt_seeds.split(",") if s.strip()}
    jobs = [
        {"family": family, "dataset": dataset, "seed": int(seed),
         "recipe": args.recipe, "force": args.force, "keep_ckpt": int(seed) in keep, **extra}
        for dataset in datasets
        for family in (f.strip() for f in args.families.split(",") if f.strip())
        for seed in (s.strip() for s in args.seeds.split(",") if s.strip())
    ]
    spawn(args.profile, jobs)
    print(f"\n{len(jobs)} job(s) spawned on {args.profile}; they run server-side. "
          f"This process can exit.", flush=True)


if __name__ == "__main__":
    main()
