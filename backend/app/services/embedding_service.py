"""
Embedding Service — Plugin Integration (sentence-transformers).

Uses the sentence-transformers library for generating document
and query embeddings. Supports loading fine-tuned models.
"""

import logging
import os
from typing import Optional
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper around sentence-transformers for embedding generation."""

    def __init__(self):
        settings = get_settings()
        model_name = settings.finetuned_model_path or settings.embedding_model

        # Check if a fine-tuned model exists on disk
        if settings.finetuned_model_path and os.path.exists(settings.finetuned_model_path):
            logger.info(f"Loading fine-tuned embedding model from: {settings.finetuned_model_path}")
            model_name = settings.finetuned_model_path
        else:
            logger.info(f"Loading base embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded — dimension: {self.dimension}")

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> list[list[float]]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string.

        Args:
            query: The query text to encode.

        Returns:
            Embedding vector for the query.
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode a batch of documents for indexing.

        Args:
            documents: List of document texts.
            batch_size: Batch size for encoding.

        Returns:
            List of embedding vectors.
        """
        return self.encode(documents, batch_size=batch_size, show_progress=True)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
        }
