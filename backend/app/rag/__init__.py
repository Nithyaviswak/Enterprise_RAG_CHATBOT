"""RAG pipeline package.

Production-oriented retrieval pipeline:
    parse → chunk → embed → vector store → retrieve → filter → rerank
    → guardrails → generate → verify → attribute sources

Each stage is a small, testable module. The end-to-end orchestrator lives in
``app.rag.pipeline``.
"""
