import json
import logging
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)

try:
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"X-Title": "InfoQuest Assessment"},
    )
    logger.info("OpenRouter LLM client initialised | model=%s", settings.llm_model)
except Exception as e:
    logger.error("Failed to initialise OpenRouter client: %s", e)
    raise


def rewrite_query(query: str, conversation_history: list[dict] = None) -> str:
    """
    Expand the user's query with synonyms and domain terms so it
    retrieves more relevant candidates from the vector DB.

    e.g. "ML engineer in Gulf" →
    "Machine Learning engineer Python TensorFlow UAE Dubai Saudi Arabia"
    """
    if conversation_history is None:
        conversation_history = []

    history_text = ""
    if conversation_history:
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}"
            for t in conversation_history
        )

    logger.debug("Rewriting query: '%s'", query)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": f"""Rewrite this candidate search query to improve vector search retrieval.
Add skill synonyms, industry terms, and the location of the candidates, language also you can lookup from the conv history if relevant.


{f"Conversation history:{chr(10)}{history_text}{chr(10)}" if history_text else ""}
Query: {query}

Rewritten:"""}],
            max_tokens=150,
            temperature=0.2,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.debug("Query rewritten | original='%s' → rewritten='%s'", query, rewritten)
        return rewritten
    except Exception as e:
        logger.error("rewrite_query LLM call failed: %s", e)
        raise


def explain_match(query: str, candidate: dict) -> dict:
    """
    Given a query and candidate metadata, return a one-sentence explanation
    and 2-3 highlights of why this person matches.
    """
    candidate_id = candidate.get("id", "unknown")
    logger.debug("Generating explanation for candidate %s", candidate_id)

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": f"""You are writing search results for an expert network.

Search query: "{query}"

Candidate:
- Name: {candidate.get('name')}
- Title: {candidate.get('current_title')} at {candidate.get('current_company')}
- Location: {candidate.get('city')}, {candidate.get('country')}
- Industry: {candidate.get('industry')}
- Years of experience: {candidate.get('years_of_experience')}
- Skills: {candidate.get('skills', '')[:200]}
- Top skills: {candidate.get('top_skills')}
- Education: {candidate.get('education')}
- Languages: {candidate.get('languages')}

Return JSON only:
{{"why_match": "<one sentence why this person matches the query>", "highlights": ["<fact 1>", "<fact 2>", "<fact 3>"]}}"""}],
            max_tokens=250,
            temperature=0.3,
        )

        raw = response.choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()

        parsed = json.loads(raw)
        logger.debug("Explanation generated for candidate %s", candidate_id)
        return parsed

    except json.JSONDecodeError as e:
        logger.warning("explain_match returned invalid JSON for candidate %s: %s", candidate_id, e)
        return _fallback_explanation(candidate)
    except Exception as e:
        logger.error("explain_match LLM call failed for candidate %s: %s", candidate_id, e)
        return _fallback_explanation(candidate)


def summarise(query: str, candidates: list[dict]) -> str:
    logger.debug("Generating summary for %d candidates", len(candidates))
    names = ", ".join(c.get("name", "") for c in candidates[:5])

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": f'Search: "{query}". Top results: {names}. Write 2 sentences summarising why these candidates are relevant. Plain text only.'}],
            max_tokens=100,
            temperature=0.4,
        )
        summary = response.choices[0].message.content.strip()
        logger.debug("Summary generated: '%s'", summary[:80])
        return summary
    except Exception as e:
        logger.error("summarise LLM call failed: %s", e)
        raise


def _fallback_explanation(candidate: dict) -> dict:
    """Return a safe fallback when the LLM fails or returns bad JSON."""
    return {
        "why_match": f"Relevant based on {candidate.get('top_skills') or candidate.get('skills') or 'experience'}.",
        "highlights": [
            candidate.get("current_title", ""),
            candidate.get("industry", ""),
            candidate.get("skills", "")[:80],
        ],
    }

def rerank_candidates(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    summaries = "\n".join(
        f"{i+1}. {c.get('name')} | {c.get('current_title')} | "
        f"{c.get('city')}, {c.get('country')} | "
        f"{c.get('years_of_experience')} yrs | "
        f"skills: {(c.get('skills') or '')[:80]}"
        for i, c in enumerate(candidates)
    )

    print(f"let's observe here ----> {summaries}\n")

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": f"""You are ranking candidates for a search query.
Pick the {top_k} best matches strictly against the user input. Return ONLY a JSON array of their numbers.
Example: [3, 1, 7, 2, 5]

Query: "{query}"

Candidates:
{summaries}

Best {top_k} (as JSON array of numbers):"""}],
        temperature=0.3,
    )
    try:
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        indices = json.loads(raw)
        return [candidates[i - 1] for i in indices if 0 < i <= len(candidates)]
    except Exception as e:
        logger.warning("Reranking failed, returning original order: %s", e)
        return candidates[:top_k]
