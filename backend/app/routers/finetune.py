"""Fine-tuning Router — Trigger and monitor embedding fine-tuning."""

import logging
from fastapi import APIRouter, Request, BackgroundTasks

from app.models.schemas import FinetuneRequest, FinetuneStatus
from app.services.finetune_service import run_finetune_pipeline, get_finetune_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/finetune", tags=["finetune"])


@router.post("/start")
async def start_finetune(
    request: Request,
    background_tasks: BackgroundTasks,
    body: FinetuneRequest = FinetuneRequest(),
):
    """Start the embedding fine-tuning pipeline.

    This runs the run-llama/finetune-embedding methodology:
    1. Generate synthetic Q&A from uploaded documents
    2. Fine-tune the embedding model
    3. Evaluate retrieval performance
    """
    current_status = get_finetune_status()
    if current_status["status"] == "running":
        return {"message": "Fine-tuning is already in progress", **current_status}

    # Run fine-tuning in background
    background_tasks.add_task(
        run_finetune_pipeline,
        corpus_dir="./data/uploads",
        base_model=body.base_model,
        epochs=body.epochs,
        batch_size=body.batch_size,
    )

    return {
        "message": "Fine-tuning pipeline started",
        "status": "running",
        "config": body.model_dump(),
    }


@router.get("/status")
async def finetune_status():
    """Check the current fine-tuning status."""
    return get_finetune_status()
