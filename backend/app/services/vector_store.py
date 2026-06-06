"""
Vector Store Service — ChromaDB Integration.

Manages document embeddings storage and semantic search
using ChromaDB as the persistent vector database.
"""

import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """ChromaDB vector store for document embeddings."""

    def __init__(self, embedding_service=None):
        settings = get_settings()
        persist_dir = settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = settings.chroma_collection_name
        self.embedding_service = embedding_service

        # Get or create the main collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized — collection '{self.collection_name}' "
            f"with {self.collection.count()} documents"
        )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ):
        """Add documents with pre-computed embeddings to the store.

        Args:
            documents: List of document text content.
            embeddings: Pre-computed embedding vectors.
            metadatas: Metadata dicts for each document.
            ids: Unique IDs for each document.
        """
        if not documents:
            return

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(documents)} documents to vector store")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Query the vector store for similar documents.

        Args:
            query_embedding: Query vector.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            List of result dicts with content, source, score, and metadata.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        if self.collection.count() == 0:
            return []

        results = self.collection.query(**kwargs)

        parsed_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                parsed_results.append({
                    "content": doc,
                    "source": results["metadatas"][0][i].get("source", "Unknown") if results["metadatas"] else "Unknown",
                    "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return parsed_results

    def delete_by_metadata(self, where: dict):
        """Delete documents matching metadata filter.

        Args:
            where: Metadata filter for deletion.
        """
        try:
            self.collection.delete(where=where)
            logger.info(f"Deleted documents matching filter: {where}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
        }

    def reset(self):
        """Reset the collection (delete all documents)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset")
