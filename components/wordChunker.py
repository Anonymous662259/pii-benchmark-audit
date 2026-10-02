"""Recursive, label-aware word chunker built on chonkie.RecursiveChunker.

The chunk-size budget is computed in the model's own word-token space (via the
same ``WhitespaceTokenSplitter`` the model uses): the entity-schema label tokens
are subtracted from the encoder context window so every emitted chunk plus the
labels fits. Recursion falls back paragraph -> sentence -> whitespace.
"""

from typing import Any, Dict, List

from config.hyperparams import CHUNK_SAFETY_MARGIN, MAX_CHARS_PER_CHUNK


def _word_counter(text: str) -> int:
    """Count word tokens the way the model's tokenizer splits them."""
    from gliner2.processor import WhitespaceTokenSplitter

    splitter = WhitespaceTokenSplitter()
    return sum(1 for _ in splitter(text, lower=False))


class RecursiveChunkerWrapper:
    """One-time-init wrapper: holds a chonkie ``RecursiveChunker`` whose chunk-size
    budget already accounts for the entity-schema label tokens.

    Each emitted chunk exposes ``text`` and ``start_index`` (chonkie ``Chunk``).
    """

    def __init__(
        self,
        budget_words: int,
        max_chars: int = MAX_CHARS_PER_CHUNK,
        min_chars_per_chunk: int = 24,
    ):
        # chonkie is imported lazily so that merely importing this module (and,
        # transitively, evaluate.py) does not hard-require the optional dep.
        from chonkie.chunker.recursive import RecursiveChunker
        from chonkie.types import RecursiveLevel, RecursiveRules

        self.budget = budget_words
        self.max_chars = max_chars
        self._chunker = RecursiveChunker(
            tokenizer=_word_counter,
            chunk_size=budget_words,
            rules=RecursiveRules(
                levels=[
                    RecursiveLevel(delimiters=["\n\n", "\r\n", "\n", "\r"]),
                    RecursiveLevel(delimiters=[". ", "! ", "? ", ".\n", "!\n", "?\n"]),
                    RecursiveLevel(whitespace=True),
                ]
            ),
            min_characters_per_chunk=min_chars_per_chunk,
        )

    def chunk(self, text: str) -> List[Any]:
        """Split ``text`` into chonkie ``Chunk`` objects (``text``/``start_index``)."""
        return self._chunker.chunk(text)


def build_word_chunker(
    chunk_tokens: int,
    label_descriptions: Dict[str, str],
    safety_margin: int = CHUNK_SAFETY_MARGIN,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> RecursiveChunkerWrapper:
    """Build a :class:`RecursiveChunkerWrapper` with a label-aware token budget.

    The label tokens (``"<LABEL> <description>"`` plus a +2 separator budget each)
    are subtracted from ``chunk_tokens`` so the encoder never overflows.
    """
    labels_words = [f"{k} {v}" for k, v in label_descriptions.items()]
    label_token_count = sum(_word_counter(lbl) + 2 for lbl in labels_words)
    budget = max(64, chunk_tokens - label_token_count - safety_margin)
    return RecursiveChunkerWrapper(budget_words=budget, max_chars=max_chars)
