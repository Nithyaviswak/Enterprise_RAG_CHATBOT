"""
RAGFlow Client — Plugin Integration.

Communicates with RAGFlow's REST API for document parsing,
knowledge base management, and retrieval.
RAGFlow runs as a separate Docker service.
"""

import logging
from typing import Optional
import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class RAGFlowClient:
    """HTTP client for RAGFlow REST API."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.ragflow_base_url.rstrip("/")
        self.api_key = settings.ragflow_api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._available = bool(self.api_key)
        if not self._available:
            logger.warning("RAGFlow API key not configured — RAGFlow features disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated request to RAGFlow API."""
        url = f"{self.base_url}/api/v1{path}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.request(
                    method, url, headers=self.headers, **kwargs
                )
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            logger.error(f"Cannot connect to RAGFlow at {self.base_url}")
            raise ConnectionError(
                f"RAGFlow is not reachable at {self.base_url}. "
                "Ensure RAGFlow is running (docker compose up -d)."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"RAGFlow API error: {e.response.status_code} — {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"RAGFlow request failed: {e}")
            raise

    # ── Dataset (Knowledge Base) Management ──────────────────

    async def create_dataset(self, name: str, description: str = "") -> dict:
        """Create a new knowledge base dataset."""
        return await self._request(
            "POST",
            "/datasets",
            json={"name": name, "description": description},
        )

    async def list_datasets(self) -> list[dict]:
        """List all datasets."""
        result = await self._request("GET", "/datasets")
        return result.get("data", [])

    async def delete_dataset(self, dataset_id: str) -> dict:
        """Delete a dataset."""
        return await self._request("DELETE", f"/datasets/{dataset_id}")

    # ── Document Management ──────────────────────────────────

    async def upload_document(self, dataset_id: str, file_path: str, filename: str) -> dict:
        """Upload a document to a dataset for parsing."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f)}
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = await client.post(
                    f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
                    files=files,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

    async def parse_document(self, dataset_id: str, document_ids: list[str]) -> dict:
        """Trigger document parsing."""
        return await self._request(
            "POST",
            f"/datasets/{dataset_id}/chunks",
            json={"document_ids": document_ids},
        )

    async def get_document_chunks(self, dataset_id: str, document_id: str) -> list[dict]:
        """Get parsed chunks for a document."""
        result = await self._request(
            "GET",
            f"/datasets/{dataset_id}/documents/{document_id}/chunks",
        )
        return result.get("data", {}).get("chunks", [])

    # ── Retrieval ─────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        dataset_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve relevant chunks using RAGFlow's retrieval engine."""
        payload = {
            "question": query,
            "top_k": top_k,
        }
        if dataset_ids:
            payload["dataset_ids"] = dataset_ids

        try:
            result = await self._request("POST", "/retrieval", json=payload)
            chunks = result.get("data", {}).get("chunks", [])
            return [
                {
                    "content": chunk.get("content", ""),
                    "source": chunk.get("document_name", "Unknown"),
                    "score": chunk.get("similarity", 0.0),
                    "metadata": {
                        "document_id": chunk.get("document_id", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                    },
                }
                for chunk in chunks
            ]
        except Exception as e:
            logger.warning(f"RAGFlow retrieval failed: {e}")
            return []

    # ── Chat (RAGFlow's built-in chat) ────────────────────────

    async def create_chat_assistant(self, name: str, dataset_ids: list[str]) -> dict:
        """Create a chat assistant in RAGFlow."""
        return await self._request(
            "POST",
            "/chats",
            json={"name": name, "dataset_ids": dataset_ids},
        )

    async def health_check(self) -> bool:
        """Check if RAGFlow is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/v1/datasets", headers=self.headers)
                return response.status_code == 200
        except Exception:
            return False
