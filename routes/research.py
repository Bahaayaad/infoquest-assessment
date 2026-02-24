import asyncio
import logging

from fastapi import APIRouter

from database.vectorstore import search
from models.research import ResearchResponse, ResearchRequest, IterationLog
from services import llm
from services.embeddings import embed_query

router  = APIRouter()
@router.post('/research', response_model=ResearchResponse)
async def research(request: ResearchRequest):
    loop = asyncio.get_event_loop()

    all_candidates = dict[str, dict] = {}
    trace: list[IterationLog] = []
    history: list[dict] = []
    iteration = 0

    while iteration < request.iterations:
        iteration = iteration + 1

        try:
            step = await loop.run_in_executor(
                None,
                llm.react_agent_step,
                request.qury,
                history,
                len(all_candidates),
                iteration,
                request.max_iteration,
            )
        except Exception as e:
            logging.error(e)
            break

        thought = step.get('thought')
        action = step.get('action')
        reasoning = step.get('reasoning')
        action_input  = step.get('action_input')

        if action == "stop":
            stop_reason = reasoning or "agent stopped reasoning"
            trace.append(IterationLog(
                iteration=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                reasoning=stop_reason,
                observation= "Agent stopped the loop",
            ))

            break

        if action == "search":
            try:
                query_vector = await loop.run_in_executor(None, embed_query, action_input)
            except Exception as e:
                pass
            else:
                raw_result = search (query_vector, top_k= 5)

                new_result = [r for r in raw_result if r['id'] not in all_candidates]
                for r in new_result:
                    all_candidates[r["id"]] = r

            if new_result:
                #TODO: Continue Implementing the route



        trace.append(IterationLog(iteration=iteration, thought=request.thought))

