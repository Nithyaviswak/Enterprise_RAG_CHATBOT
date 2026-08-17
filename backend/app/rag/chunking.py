"""Configurable document chunking.

Two strategies are provided:

- ``chunk_by_words``: token (word) based splitting with overlap. Simple,
  deterministic, and language-neutral.
- ``chunk_recursive``: recursive splitter on natural boundaries
  (paragraph → sentence → word) for higher-quality units.

Both are pure functions so they are easy to unit test. Sizes and overlap are
driven by :class:`app.config.Settings` rather than hard-coded constants.
"""

import re
from typing import Callable, Optional

from app.config import get_settings

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def chunk_by_words(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_length: Optional[int] = None,
) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Source text to chunk.
        chunk_size: Words per chunk (default: settings.chunk_size).
        overlap: Overlapping words between adjacent chunks
            (default: settings.chunk_overlap).
        min_length: Drop chunks with fewer words than this
            (default: settings.chunk_min_length).

    Returns:
        List of chunk strings.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    min_length = min_length if min_length is not None else settings.chunk_min_length

    if overlap >= chunk_size:
        overlap = max(chunk_size // 4, 1)

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= min_length:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_recursive(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_length: Optional[int] = None,
) -> list[str]:
    """Split text into chunks preferring natural boundaries.

    Tries paragraphs first, then sentences, then falls back to words. Chunks
    larger than ``chunk_size`` words continue down the fallback chain.

    Args:
        text: Source text to chunk.
        chunk_size: Target words per chunk.
        overlap: Overlapping words between chunks.
        min_length: Minimum words for a kept chunk.

    Returns:
        List of chunk strings.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    min_length = min_length if min_length is not None else settings.chunk_min_length

    normalized = re.sub(r"[ \t]+", " ", text)
    units = [u.strip() for u in _PARAGRAPH_BOUNDARY.split(normalized) if u.strip()]

    return _merge_units(units, chunk_size, overlap, min_length)


def _merge_units(
    units: list[str],
    chunk_size: int,
    overlap: int,
    min_length: int,
) -> list[str]:
    """Merge atomic units into chunks of roughly ``chunk_size`` words."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    def flush():
        nonlocal buffer, buffer_words
        if buffer:
            joined = " ".join(buffer).strip()
            if len(joined.split()) >= min_length:
                chunks.append(joined)
            # keep last ``overlap`` words for continuity
            keep = buffer
            while len(" ".join(keep).split()) > overlap and len(keep) > 1:
                keep.pop(0)
            buffer = keep
            buffer_words = len(" ".join(keep).split())

    for unit in units:
        for sentence in [s.strip() for s in _SENTENCE_BOUNDARY.split(unit) if s.strip()]:
            sentence_words = sentence.split()
            if buffer_words + len(sentence_words) <= chunk_size:
                buffer.append(sentence)
                buffer_words += len(sentence_words)
            else:
                flush()
                if len(sentence_words) > chunk_size:
                    for sub in chunk_by_words(sentence, chunk_size, overlap, 4):
                        sub_words = sub.split()
                        if buffer and buffer_words + len(sub_words) > chunk_size:
                            flush()
                        buffer.append(sub)
                        buffer_words += len(sub_words)
                else:
                    buffer.append(sentence)
                    buffer_words = len(sentence_words)
    flush()
    return chunks


def get_chunkers() -> dict[str, Callable[..., list[str]]]:
    """Return the available chunking strategies."""
    return {
        "words": chunk_by_words,
        "recursive": chunk_recursive,
    }