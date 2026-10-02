#!/usr/bin/env python3
"""Zero-shot GLiNER v1 (pip `gliner`) on a registry corpus, dumped in evaluate.py's schema.

PROTOCOL.md 9.10: off-the-shelf PII models evaluated without training.  Rules fixed before
the runs:
  * label text = the type's registry description (the natural-language label a GLiNER v1
    model reads), mapped back to the type; a description shared by two types gets the type
    code appended so the mapping stays one-to-one
  * documents are cut at whitespace into chunks of <= --chunk-words words, types into groups
    of <= --label-group, so neither the text nor the label prompt is truncated by max_len
  * every (chunk, group) is decoded at --threshold with flat_ner; the union is then decoded
    flat and greedily by score (non-overlapping across all types), as evaluate.py does
  * an exception on a document is recorded in its `error` field (the Modal runner fails the
    run if any document has one)

    python scripts/eval_gliner_zeroshot.py --model urchade/gliner_multi_pii-v1 \
        --ner-dataset kiji --split test --output-dir out/kiji/test --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_WORD = re.compile(r"\S+")


def _token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    p = Path(__file__).resolve().parents[1] / ".hf_token"
    return tok or (p.read_text(encoding="utf-8").strip() if p.is_file() else "")


def label_texts(descriptions: Dict[str, str]) -> Dict[str, str]:
    """type -> label text (its description; the type code is appended where it collides)."""
    counts = Counter(descriptions.values())
    return {t: (d if counts[d] == 1 else f"{d} ({t})") for t, d in descriptions.items()}


def chunks(text: str, max_words: int) -> List[Tuple[int, int]]:
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    return [(words[i][0], words[min(i + max_words, len(words)) - 1][1])
            for i in range(0, len(words), max_words)]


def greedy_flat(preds: List[dict]) -> List[dict]:
    kept: List[dict] = []
    for p in sorted(preds, key=lambda p: -p["score"]):
        if all(p["end"] <= k["start"] or k["end"] <= p["start"] for k in kept):
            kept.append(p)
    return sorted(kept, key=lambda p: (p["start"], p["end"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ner-dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk-words", type=int, default=200)
    ap.add_argument("--label-group", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from gliner import GLiNER
    from config.ner_datasets import NER_DATASETS

    cfg = NER_DATASETS[args.ner_dataset]
    data = load_dataset(cfg["repo"], split=args.split, token=_token() or None)
    if args.max_samples:
        data = data.select(range(min(args.max_samples, len(data))))
    to_label = label_texts(cfg["label_descriptions"])
    to_type = {v: k for k, v in to_label.items()}
    names = list(to_label.values())
    groups = [names[i:i + args.label_group] for i in range(0, len(names), args.label_group)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(args.model, map_location=device)
    model.eval()

    # one inference call per label group over every chunk of every document
    docs = [r["source_text"] for r in data]
    spans = [chunks(t, args.chunk_words) for t in docs]
    flat = [(di, s, e) for di, cs in enumerate(spans) for s, e in cs]
    per_doc: List[List[dict]] = [[] for _ in docs]
    errors: Dict[int, str] = {}
    started = time.time()
    for group in groups:
        for lo in range(0, len(flat), 256):          # bounded memory per call
            block = flat[lo:lo + 256]
            texts = [docs[di][s:e] for di, s, e in block]
            try:
                with torch.no_grad():
                    out = model.inference(texts=texts, labels=group, threshold=args.threshold,
                                          flat_ner=True, multi_label=False, batch_size=args.batch_size)
            except Exception as exc:                  # retry one chunk at a time, record failures
                out = []
                for (di, s, e), t in zip(block, texts):
                    try:
                        with torch.no_grad():
                            out.append(model.inference(texts=[t], labels=group, threshold=args.threshold,
                                                       flat_ner=True, multi_label=False, batch_size=1)[0])
                    except Exception as exc2:
                        errors[di] = f"{type(exc2).__name__}: {exc2}"[:300]
                        out.append([])
                    if device == "cuda":
                        torch.cuda.empty_cache()
            for (di, s, _), ents in zip(block, out):
                for ent in ents:
                    per_doc[di].append({"label": to_type[ent["label"]], "start": s + ent["start"],
                                        "end": s + ent["end"], "score": float(ent["score"])})

    out_dir = Path(args.output_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "all.jsonl").open("w", encoding="utf-8") as sink:
        for i, (row, text) in enumerate(zip(data, docs)):
            preds = greedy_flat(per_doc[i])
            for p in preds:
                p["text"] = text[p["start"]:p["end"]]
            gt = [{"start": m["start"], "end": m["end"], "tag": m["label"], "value": m["value"],
                   "text": text[m["start"]:m["end"]]} for m in row["privacy_mask"] or []]
            sink.write(json.dumps({"sample_id": str(i), "language": row.get("language", "unknown"),
                                   "text": text, "gt": gt, "predictions": preds,
                                   "error": errors.get(i)}, ensure_ascii=False) + "\n")
    info = {"model": args.model, "dataset": cfg["repo"], "split": args.split, "n_samples": len(docs),
            "threshold": args.threshold, "chunk_words": args.chunk_words, "label_group": args.label_group,
            "label_texts": to_label, "failed_documents": len(errors), "seconds": round(time.time() - started, 1)}
    (Path(args.output_dir) / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    print(f"{args.model} on {args.ner_dataset}/{args.split}: {len(docs)} docs, "
          f"{sum(len(p) for p in per_doc)} raw spans, {len(errors)} failed, {info['seconds']}s")


if __name__ == "__main__":
    main()
