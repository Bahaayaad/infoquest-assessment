import uuid
import asyncio
import logging
from fastapi import APIRouter, HTTPException

from models.candidate import CandidateResult
from models.chat import ChatResponse, ChatRequest
from services.embeddings import embed_query
from database.vectorstore import search, count
from services import llm

logger = logging.getLogger(__name__)
router = APIRouter()
conversations: dict[str, list[dict]] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Search for candidates using natural language.
    Pass a conversation_id to continue a previous session.
    """
    logger.info("Chat request | query='%s' conversation_id=%s", request.query, request.conversation_id)

    if count() == 0:
        logger.warning("Chat called but vector store is empty")
        raise HTTPException(status_code=503, detail="No candidates indexed yet. Run POST /ingest first.")

    loop = asyncio.get_event_loop()

    cid = request.conversation_id or str(uuid.uuid4())
    if cid not in conversations:
        conversations[cid] = []
        logger.info("New conversation created | conversation_id=%s", cid)
    else:
        logger.info("Continuing conversation | conversation_id=%s turns=%d", cid, len(conversations[cid]))

    history = conversations[cid]

    try:
        rewritten = await loop.run_in_executor(
            None, llm.rewrite_query, request.query, history
        )
        logger.info("Query rewritten | original='%s' rewritten='%s'", request.query, rewritten)
    except Exception as e:
        logger.warning("Query rewrite failed, using original query: %s", e)
        rewritten = request.query

    try:
        query_vector = await loop.run_in_executor(None, embed_query, rewritten)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        raw_results = search(query_vector, top_k=request.top_k * 4)
        results = await loop.run_in_executor(
            None, llm.rerank_candidates, request.query, raw_results, request.top_k
        )
        logger.info("Vector search returned %d results", len(results))
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    if not results:
        logger.info("No results found for query='%s'", request.query)
        return ChatResponse(
            conversation_id=cid,
            query=request.query,
            candidates=[],
            summary="No matching candidates found. Try broadening your search.",
        )

    semaphore = asyncio.Semaphore(3)
    async def explain_one(r: dict) -> dict:
        async with semaphore:
            try:
                explanation = await loop.run_in_executor(
                    None, llm.explain_match, request.query, r
                )
                return {**r, **explanation}
            except Exception as e:
                logger.warning("explain_match failed for candidate %s: %s", r.get("id"), e)
                return {
                    **r,
                    "why_match": f"Relevant based on {r.get('top_skills') or r.get('skills') or 'experience'}.",
                    "highlights": [r.get("current_title", ""), r.get("industry", ""), r.get("skills", "")[:80]],
                }

    enriched = await asyncio.gather(*[explain_one(r) for r in results])
    logger.info("Explanations generated for %d candidates", len(enriched))

    try:
        summary = await loop.run_in_executor(
            None, llm.summarise, request.query, list(enriched)
        )
    except Exception as e:
        logger.warning("Summary generation failed: %s", e)
        summary = f"Found {len(enriched)} candidates matching your search."

    candidates = [
        CandidateResult(
            id=r["id"],
            name=r.get("name", ""),
            headline=r.get("headline") or None,
            current_title=r.get("current_title") or None,
            current_company=r.get("current_company") or None,
            location=", ".join(filter(None, [r.get("city"), r.get("country")])) or None,
            industry=r.get("industry") or None,
            years_of_experience=r.get("years_of_experience") or None,
            skills=r.get("skills") or None,
            languages=r.get("languages") or None,
            education=r.get("education") or None,
            relevance_score=r["score"],
            why_match=r.get("why_match", ""),
            highlights=[h for h in r.get("highlights", []) if h],
        )
        for r in enriched
    ]

    conversations[cid].append({"role": "user", "content": request.query})
    conversations[cid].append({"role": "assistant", "content": summary})

    logger.info("Chat complete | conversation_id=%s candidates_returned=%d", cid, len(candidates))

    return ChatResponse(
        conversation_id=cid,
        query=request.query,
        candidates=candidates,
        summary=summary,
    )