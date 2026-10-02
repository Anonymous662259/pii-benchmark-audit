#!/usr/bin/env python3
"""Chunk TAB's train/validation documents for training (PROTOCOL.md 9.15).

GLiNER2's trainer truncates every text at max_len = 2048 words; 200 of TAB's 1,014
training judgments are longer, which would silently drop 9,963 of 87,459 training spans.
This splits train and validation documents into pieces of at most --max-words words,
cutting at a line break where possible and never inside a gold span, so no span is lost.
The test split is copied unchanged, so every system is still scored on the same TAB test.

    python scripts/chunk_tab.py               # check only
    python scripts/chunk_tab.py --push        # -> anonymous-org/pii-tab-chunked (private)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

_WORD = re.compile(r"\S+")


def chunk(row: Dict, max_words: int) -> List[Dict]:
    text, spans = row["source_text"], sorted(row["privacy_mask"], key=lambda m: m["start"])
    words = list(_WORD.finditer(text))
    if len(words) <= max_words:
        return [row]
    out, start_char, i = [], 0, 0
    while i < len(words):
        j = min(i + max_words, len(words))
        if j < len(words):
            cut = words[j].start()
            # prefer the last line break inside the window
            nl = text.rfind("\n", words[i].start(), cut)
            if nl > words[i].start():
                cut = nl + 1
            # never cut inside a span: move the cut before any span that straddles it
            for m in spans:
                if m["start"] < cut < m["end"]:
                    cut = m["start"]
            if cut <= start_char:                      # degenerate: fall back to the word cut
                cut = words[j].start()
        else:
            cut = len(text)
        piece = text[start_char:cut]
        mask = [{**m, "start": m["start"] - start_char, "end": m["end"] - start_char}
                for m in spans if m["start"] >= start_char and m["end"] <= cut]
        if piece.strip():
            out.append({**row, "source_text": piece, "privacy_mask": mask})
        start_char = cut
        while i < len(words) and words[i].start() < cut:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-words", type=int, default=1000)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    from datasets import Dataset, DatasetDict, load_dataset
    token = Path(".hf_token").read_text(encoding="utf-8").strip()
    built = {}
    for split in ("train", "validation", "test"):
        rows = list(load_dataset("anonymous-org/pii-tab", split=split, token=token))
        if split == "test":
            built[split] = rows
        else:
            built[split] = [c for r in rows for c in chunk(r, args.max_words)]
        before = sum(len(r["privacy_mask"]) for r in rows)
        after = sum(len(r["privacy_mask"]) for r in built[split])
        bad = sum(1 for r in built[split] for m in r["privacy_mask"]
                  if r["source_text"][m["start"]:m["end"]] != m["value"])
        longest = max(len(_WORD.findall(r["source_text"])) for r in built[split])
        print(f"{split:10s} docs {len(rows)} -> {len(built[split])}, spans {before} -> {after}, "
              f"offset mismatches {bad}, longest {longest} words")
        assert after == before and bad == 0, "chunking lost or broke spans"
    if args.push:
        DatasetDict({k: Dataset.from_list(v) for k, v in built.items()}).push_to_hub(
            "anonymous-org/pii-tab-chunked", private=True, token=token)
        print("pushed anonymous-org/pii-tab-chunked")


if __name__ == "__main__":
    main()
