import asyncio
import logging
from fastapi import APIRouter, HTTPException

from models.research import ResearchRequest, ResearchResponse, IterationLog
from models.candidate import CandidateResult
from services.embeddings import embed_query
from database.vectorstore import search, count
from services import llm

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):

    logger.info("ReAct research started | query='%s'", request.query)

    if count() == 0:
        raise HTTPException(status_code=503, detail="No candidates indexed. Run POST /ingest first.")

    loop = asyncio.get_event_loop()

    all_candidates: dict[str, dict] = {}
    trace: list[IterationLog] = []
    history: list[dict] = []
    stop_reason = "max_iterations_reached"
    iteration = 0

    while iteration < request.max_iterations:
        iteration += 1
        logger.info("ReAct iteration %d | total_collected=%d", iteration, len(all_candidates))

        try:
            step = await loop.run_in_executor(
                None,
                llm.react_agent_step,
                request.query,
                history,
                len(all_candidates),
                iteration,
                request.max_iterations,
            )
        except Exception as e:
            logger.error("react_agent_step failed: %s", e)
            stop_reason = "agent_error"
            break

        thought = step.get("thought", "")
        action = step.get("action", "stop")
        action_input = step.get("action_input", "")

        logger.info("THOUGHT: %s", thought)
        logger.info("ACTION: %s | INPUT: %s", action, action_input)

        if action == "stop":
            stop_reason = action_input or "agent_decided_to_stop"

            trace.append(IterationLog(
                iteration=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                observation="Agent decided to stop the search loop.",
            ))

            history.append({"role": "THOUGHT", "content": thought})
            history.append({"role": "ACTION", "content": f"stop({action_input})"})
            history.append({"role": "OBSERVATION", "content": "Search stopped."})

            logger.info("Agent stopped | reason=%s", stop_reason)
            break

        elif action == "search":
            try:
                query_vector = await loop.run_in_executor(None, embed_query, action_input)
            except Exception as e:
                logger.error("Embedding failed at iteration %d: %s", iteration, e)
                observation = "Embedding failed — could not execute search."
                new_count = 0
            else:
                raw_results = search(query_vector, top_k=20)

                new_results = [r for r in raw_results if r["id"] not in all_candidates]
                for r in new_results:
                    all_candidates[r["id"]] = r

                new_count = len(new_results)
                logger.info("Search returned %d new candidates | total=%d", new_count, len(all_candidates))

                if new_results:
                    names = "; ".join(
                        f"{r.get('name')} ({r.get('current_title')}, {r.get('city')}, {r.get('country')})"
                        for r in new_results[:5]
                    )
                    observation = f"Found {new_count} new candidates: {names}. Total collected: {len(all_candidates)}."
                else:
                    observation = f"No new candidates found. Total collected: {len(all_candidates)}."

            logger.info("OBSERVATION: %s", observation)

            trace.append(IterationLog(
                iteration=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            ))

            history.append({"role": "THOUGHT", "content": thought})
            history.append({"role": "ACTION", "content": f"search({action_input})"})
            history.append({"role": "OBSERVATION", "content": observation})

            if len(all_candidates) >= request.min_results:
                stop_reason = "sufficient_results"
                logger.info("Early stop — sufficient results (%d)", len(all_candidates))
                break

    all_list = list(all_candidates.values())

    if all_list:
        try:
            final_results = await loop.run_in_executor(
                None,
                llm.rerank_candidates,
                request.query,
                all_list,
                min(len(all_list), 10)
            )
        except Exception as e:
            logger.warning("Final rerank failed, sorting by score: %s", e)
            final_results = sorted(all_list, key=lambda r: r.get("score", 0), reverse=True)[:10]
    else:
        final_results = []

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
            relevance_score=r.get("score", 0.0),
            why_match=r.get("why_match", ""),
            highlights=[],
        )
        for r in final_results
    ]

    logger.info(
        "ReAct research complete | iterations=%d total=%d final=%d stop=%s",
        iteration, len(all_candidates), len(candidates), stop_reason
    )

    return ResearchResponse(
        query=request.query,
        candidates=candidates,
        total_found=len(all_candidates),
        iterations_ran=iteration,
        stop_reason=stop_reason,
        react_trace=trace,
    )