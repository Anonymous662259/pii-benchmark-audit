#!/usr/bin/env python3
"""Pull finished PROTOCOL.md runs (record.json + prediction dumps only) into out_pii_baselines/.

Unlike pull_pii_results.py this never downloads a run's `checkpoint/` (1.2 GB each with
--keep-ckpt-seeds), and it only asks for runs whose record says status=ok.

    python scripts/pull_rare_pii.py                       # batch 1 of PROTOCOL.md
    python scripts/pull_rare_pii.py --watch 300           # repeat every 5 minutes until complete
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

VOLUME = "pii-baselines-results"
ROOT = Path("out_pii_baselines")
# (protocol, profile, dataset, seed) per batch of PROTOCOL.md section 10
BATCHES = {
    1: [("v1-bce-bundle-50k", p, ds, s)
        for p, dss in {"<workspace>": ["gretel", "kiji"], "<workspace>": ["privy", "pii_500k"]}.items()
        for ds in dss for s in (42, 1337)],
    # seed 2024 for (b) on all four corpora; the (c) seed-2024 runs were cancelled
    # (PROTOCOL.md section 10, 2026-09-27 ~15:30)
    2: [("v1-bce-bundle-50k", p, ds, 2024)
        for p, dss in {"<workspace>": ["gretel", "pii_500k"], "<workspace>": ["kiji", "privy"]}.items()
        for ds in dss],
    # PROTOCOL.md 9.7: zero-shot GLiNER2 on the ten audited corpora (workspace <workspace>)
    3: [("v1-zeroshot", "<workspace>", ds, 0)
        for ds in ("privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k",
                   "gretel_finance", "tab", "mapa")],
    # PROTOCOL.md 9.8: (a) and (b) trained on the two heavy-artifact corpora (<workspace>)
    4: [(proto, "<workspace>", ds, s)
        for proto in ("v1-bce-baseline-50k", "v1-bce-bundle-50k")
        for ds in ("ai4p_300k", "ai4p_400k") for s in (42, 1337)],
    # PROTOCOL.md 9.9: ablation arm (a') = bce + full grid, seed 42
    # PROTOCOL.md 9.10: four released PII checkpoints, zero-shot on the ten corpora
    6: [(f"v1-zeroshot-{slug}", profile, ds, 0, family)
        for slug, profile, family in (("gliner2-privacy-filter-pii-multi", "<workspace>", "gliner2"),
                                      ("gliner_multi_pii-v1", "<workspace>", "gliner"),
                                      ("gliner-pii", "<workspace>", "gliner"),
                                      ("gretel-gliner-bi-large-v1.0", "<workspace>", "gliner"))   # moved from <workspace>: stale deploy there
        for ds in ("privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k",
                   "gretel_finance", "tab", "mapa")],
    # PROTOCOL.md 9.12 (+ the 400k (b) checkpoint run of 9.11, tagged v3), workspace <workspace>
    7: [(proto, "<workspace>", "ai4p_300k", s) for proto in ("v1-bce-baseline-50k", "v1-bce-bundle-50k")
        for s in (42, 1337)] + [("v3-bce-bundle-50k", "<workspace>", "ai4p_400k", 42)],
    # PROTOCOL.md 9.11: the four single-corpus (b) checkpoints on all ten corpora, <workspace>
    8: [(f"v1-zeroshot-pii-ckpt-bundle-{home}-s42", "<workspace>", ds, 0, "gliner2")
        for home in ("privy", "pii_500k", "kiji", "gretel")
        for ds in ("privy", "pii_500k", "kiji", "gretel", "nemotron_pii", "ai4p_300k", "ai4p_400k",
                   "gretel_finance", "tab", "mapa")],
    # PROTOCOL.md 9.11 addendum: (b) seed 42 homes on Nemotron-PII and Gretel-Finance ("<workspace>")
    10: [("v1-bce-bundle-50k", "<workspace>", ds, 42) for ds in ("nemotron_pii", "gretel_finance")],
    # PROTOCOL.md 9.13: ablation (a') seed 1337, <workspace>
    9: [("v1-bce-fullgrid-50k", "<workspace>", ds, 1337) for ds in ("privy", "pii_500k", "kiji", "gretel")],
    5: [("v1-bce-fullgrid-50k", p, ds, 42)
        for p, dss in {"<workspace>": ["privy", "gretel"], "<workspace>": ["kiji", "pii_500k"]}.items()
        for ds in dss],
}


def modal_get(profile: str, remote: str, local: Path) -> bool:
    local.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(["modal", "volume", "get", VOLUME, remote, str(local), "--force"],
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    # `modal volume get` has returned exit 0 with "Not found" before; trust the text too.
    return proc.returncode == 0 and "Not found" not in proc.stdout + proc.stderr


def pull_run(profile: str, protocol: str, ds: str, seed: int, family: str = "gliner2") -> str:
    rel = f"{protocol}/{family}/{ds}/seed-{seed}"
    dest = ROOT / rel
    record = dest / "record.json"
    if record.is_file() and json.loads(record.read_text(encoding="utf-8")).get("status") == "ok" \
            and (dest / "test" / "predictions" / "all.jsonl").is_file():
        return "have"
    with tempfile.TemporaryDirectory() as tmp:
        if not modal_get(profile, f"/{rel}/record.json", Path(tmp)):
            return "not finished"
        rec = json.loads((Path(tmp) / "record.json").read_text(encoding="utf-8"))
    if rec.get("status") != "ok":
        return f"status={rec.get('status')} {rec.get('error', '')[:80]}"
    for split in rec.get("splits") or ("validation", "test"):   # zero-shot round: test only
        if not modal_get(profile, f"/{rel}/{split}", dest):
            return f"failed to download {split}"
    dest.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    return "pulled"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, default=1, choices=sorted(BATCHES))
    ap.add_argument("--watch", type=int, default=0, help="seconds between passes; 0 = once")
    args = ap.parse_args()
    while True:
        states = {}
        for entry in BATCHES[args.batch]:
            protocol, profile, ds, seed = entry[:4]
            family = entry[4] if len(entry) > 4 else "gliner2"
            key = (protocol, ds, seed)
            states[key] = pull_run(profile, protocol, ds, seed, family)
            print(f"{profile:12s} {protocol:22s} {ds:9s} seed={seed:<5d} {states[key]}", flush=True)
        done = all(s in ("have", "pulled") for s in states.values())
        failed = [k for k, s in states.items() if s.startswith(("status=", "failed"))]
        # keep pulling the other runs after one fails; stop once every run has an outcome
        settled = all(s in ("have", "pulled") or s.startswith(("status=", "failed")) for s in states.values())
        if done or settled or not args.watch:
            print("COMPLETE" if done else ("FAILED: " + str(failed) if failed else "partial"), flush=True)
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
