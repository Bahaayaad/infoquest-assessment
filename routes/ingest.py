import asyncio
import logging
from fastapi import APIRouter, HTTPException
from database.postgres import fetch_all_candidates
from models.ingest import IngestRequest, IngestResponse
from services.embeddings import build_candidate_text, embed_texts
from database.vectorstore import upsert_candidates, wipe

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest = IngestRequest()):
    """
    Pull all candidates from Postgres embed them and store in ChromaDB.
    """
    logger.info("Ingest started | force_reingest=%s", request.force_reingest)
    loop = asyncio.get_event_loop()

    if request.force_reingest:
        logger.info("Wiping existing vector store")
        try:
            wipe()
        except Exception as e:
            logger.error("Failed to wipe vector store: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to wipe vector store: {e}")

    try:
        candidates = await loop.run_in_executor(None, fetch_all_candidates)
    except Exception as e:
        logger.error("Failed to fetch candidates from Postgres: %s", e)
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    if not candidates:
        logger.warning("No candidates found in database")
        raise HTTPException(status_code=404, detail="No candidates found in database")

    logger.info("Fetched %d candidates from Postgres", len(candidates))

    BATCH = 64
    total = 0
    failed = 0

    for i in range(0, len(candidates), BATCH):
        batch = candidates[i : i + BATCH]
        try:
            texts = [build_candidate_text(c) for c in batch]
            vectors = await loop.run_in_executor(None, embed_texts, texts)
            upsert_candidates(batch, vectors)
            total += len(batch)
            logger.info("Ingested %d / %d candidates", total, len(candidates))
        except Exception as e:
            failed += len(batch)
            logger.error("Failed to embed/upsert batch at index %d: %s", i, e)

    logger.info("Ingest complete | processed=%d failed=%d", total, failed)

    return IngestResponse(
        status="done" if failed == 0 else "partial",
        total_processed=total,
        message=f"Indexed {total} candidates successfully." if failed == 0
                else f"Indexed {total} candidates. {failed} failed — check logs.",
    )