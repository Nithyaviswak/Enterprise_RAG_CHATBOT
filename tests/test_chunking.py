"""Unit tests for chunking strategies."""

from app.rag.chunking import chunk_by_words, chunk_recursive, get_chunkers

TEXT = ("word " * 1000).strip()


def test_chunk_by_words_size_and_overlap():
    chunks = chunk_by_words(TEXT, chunk_size=200, overlap=40, min_length=10)
    assert chunks
    sizes = [len(c.split()) for c in chunks]
    assert sizes[0] <= 200
    assert all(len(c.split()) >= 10 for c in chunks)


def test_chunk_by_words_overlap_contains_bridge():
    chunks = chunk_by_words(TEXT, chunk_size=100, overlap=20, min_length=5)
    assert len(chunks) > 1
    a, b = chunks[0], chunks[1]
    overlap_words = set(a.split()) & set(b.split())
    assert overlap_words, "consecutive chunks should share overlapping words"


def test_chunk_recursive_splits_on_delimiters():
    text = "Paragraph one with enough words to be long enough. Paragraph two here too. " * 20
    chunks = chunk_recursive(text, chunk_size=60, overlap=10, min_length=5)
    assert chunks
    assert all(len(c.split()) >= 5 for c in chunks)


def test_short_text_single_chunk():
    chunks = chunk_by_words("short text", chunk_size=200, overlap=40, min_length=2)
    assert chunks == ["short text"]


def test_get_chunkers_registered():
    chunkers = get_chunkers()
    assert "words" in chunkers
    assert "recursive" in chunkers
