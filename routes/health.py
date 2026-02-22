import asyncio
import logging
from fastapi import APIRouter
from database.postgres import count_candidates
from database.vectorstore import count
from models.health import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()



@router.get("/health", response_model=HealthResponse)
async def health():
    loop = asyncio.get_event_loop()
    try:
        in_db = await loop.run_in_executor(None, count_candidates)
        db_ok = True
        logger.debug("Health check | postgres=ok candidates_in_db=%d", in_db)
    except Exception as e:
        logger.error("Health check — Postgres unreachable: %s", e)
        in_db = 0
        db_ok = False

    indexed = count()

    return HealthResponse(
        status="ok" if db_ok else "db_error",
        candidates_in_db=in_db,
        candidates_indexed=indexed,
    )
