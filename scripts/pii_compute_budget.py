# -*- coding: utf-8 -*-
"""Compute budget for the PII paper (ARR checklist C): GPU hours per GPU type.

Reads every record.json on the `pii-baselines-results` Volume of every Modal workspace that
ran PII jobs (one profile per workspace, from out_pii_baselines/spawned_calls.json), keeps one
record per run directory, and sums `elapsed_seconds` by `gpu`. Smoke runs are reported apart.
Records are cached under out_pii_baselines/compute_records/ so the sum can be re-run offline.

    python scripts/pii_compute_budget.py            # fetch + summarise
    python scripts/pii_compute_budget.py --offline  # summarise the cache only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

CACHE = Path("out_pii_baselines/compute_records")
OUT = Path("out_pii_baselines/pii_compute_budget.json")
VOLUME = "pii-baselines-results"


def workspaces() -> dict:
    """workspace -> one profile that points at it."""
    calls = json.loads(Path("out_pii_baselines/spawned_calls.json").read_text(encoding="utf-8"))
    profiles = sorted({c["profile"] for c in calls if c.get("profile")})
    out = subprocess.run(["modal", "profile", "list", "--json"], capture_output=True, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    table = json.loads(out.stdout)
    ws_of = {row["name"]: row["workspace"] for row in table}
    ws = {}
    for p in profiles:
        if p in ws_of:
            ws.setdefault(ws_of[p], p)
    return ws


def fetch_one(workspace: str) -> None:
    """Runs in a child process whose MODAL_PROFILE is already set (modal reads it at import)."""
    import modal
    try:
        vol = modal.Volume.from_name(VOLUME)
        entries = [e for e in vol.listdir("/", recursive=True) if e.path.endswith("record.json")]
    except Exception as exc:                         # workspace without the volume, or no access
        print(f"{workspace:24s}: skipped, {type(exc).__name__}: {str(exc)[:80]}")
        return
    n = 0
    for e in entries:
        dst = CACHE / workspace / e.path
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"".join(vol.read_file(e.path)))
        n += 1
    print(f"{workspace:24s}: {n} records")


def fetch(ws: dict) -> None:
    import sys
    for workspace, profile in ws.items():
        env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        subprocess.run([sys.executable, __file__, "--fetch-one", workspace], env=env)


def summarise() -> dict:
    runs = {}
    for f in CACHE.rglob("record.json"):
        rel = f.relative_to(CACHE)
        workspace, run_dir = rel.parts[0], "/".join(rel.parts[1:-1])
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not r.get("elapsed_seconds") or not r.get("gpu"):
            continue
        # the same run directory on two workspaces (a rerun) counts once per workspace: both used GPU time
        runs[(workspace, run_dir)] = r
    by_gpu, smoke, by_kind = defaultdict(float), defaultdict(float), defaultdict(float)
    n_runs = defaultdict(int)
    for (ws, run_dir), r in runs.items():
        h = r["elapsed_seconds"] / 3600
        tag = run_dir.split("/")[0]
        if tag.startswith("smoke") or r.get("smoke"):
            smoke[r["gpu"]] += h
            continue
        by_gpu[r["gpu"]] += h
        n_runs[r["gpu"]] += 1
        kind = "evaluation only" if r.get("recipe") == "zeroshot" else "training + evaluation"
        by_kind[kind] += h
    res = {"gpu_hours": {g: round(h, 1) for g, h in sorted(by_gpu.items())},
           "runs": dict(n_runs),
           "total_gpu_hours": round(sum(by_gpu.values()), 1),
           "by_kind": {k: round(v, 1) for k, v in by_kind.items()},
           "smoke_gpu_hours": {g: round(h, 2) for g, h in smoke.items()},
           "records": len(runs)}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fetch-one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.fetch_one:
        fetch_one(args.fetch_one)
        return
    if not args.offline:
        fetch(workspaces())
    print(json.dumps(summarise(), indent=1))


if __name__ == "__main__":
    main()
