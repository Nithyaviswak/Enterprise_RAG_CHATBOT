"""Unit tests for ingestion (parsing + dedup + chunk building)."""

import pytest

from app.rag.failures import InvalidDocumentError
from app.rag.ingestion import (
    ALLOWED_EXTENSIONS,
    Page,
    ParsedDocument,
    build_chunks,
    content_hash,
    is_allowed,
    is_duplicate_in_store,
    parse_document,
)

TXT_SAMPLE = """Nithyananda Chari R is an AI/ML engineer based in Bangalore.
He is pursuing a B.Tech in AI and Machine Learning at Siddartha Institute
of Science and Technology."""
# repeat to force multiple chunks
TXT_LONG = (TXT_SAMPLE + "\n\n") * 40


class FakeCollection:
    def __init__(self, hashes):
        self._hashes = set(hashes)

    def get(self, where=None, limit=None):
        key = (where or {}).get("content_hash")
        if key and key in self._hashes:
            return {"ids": ["existing"], "documents": [""], "metadatas": [{}]}
        return {"ids": [], "documents": [], "metadatas": []}


def test_content_hash_stable():
    assert content_hash("hello world") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello there")


def test_is_allowed():
    assert is_allowed("resume.pdf")
    assert is_allowed("notes.docx")
    assert is_allowed("README.md")
    assert not is_allowed("malware.exe")
    assert not is_allowed("notes.zip")


def test_extensions_registered():
    assert ".pdf" in ALLOWED_EXTENSIONS
    assert ".docx" in ALLOWED_EXTENSIONS


def test_parse_document_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(TXT_SAMPLE, encoding="utf-8")
    doc = parse_document(str(path), "txt")
    assert isinstance(doc, ParsedDocument)
    assert doc.pages
    assert all(isinstance(p, Page) for p in doc.pages)
    assert doc.full_text == TXT_SAMPLE


def test_parse_document_unsupported_raises(tmp_path):
    path = tmp_path / "notes.exe"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(InvalidDocumentError):
        parse_document(str(path), "exe")


def test_parse_document_empty_raises(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")
    with pytest.raises(InvalidDocumentError):
        parse_document(str(path), "txt")


def test_build_chunks_provenance(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(TXT_LONG, encoding="utf-8")
    doc = parse_document(str(path), "txt")
    doc_id = content_hash(TXT_LONG)
    chunks = build_chunks(doc, document_id=doc_id, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    first = chunks[0]
    assert first["metadata"]["source"] == "notes.txt"
    assert first["metadata"]["document_id"] == doc_id
    assert first["metadata"]["file_type"] == ".txt"
    assert int(first["metadata"]["chunk_index"]) >= 0
    assert first["metadata"]["content_hash"]
    assert first["id"].startswith(doc_id)
    assert all(c["content"].split() for c in chunks)


def test_is_duplicate_in_store():
    doc_id = "doc-1234"
    h = content_hash(TXT_SAMPLE)
    store = FakeCollection({h})
    assert is_duplicate_in_store(store, doc_id, h)
    assert not is_duplicate_in_store(store, doc_id, content_hash("different content"))