import logging
from config import settings
from langchain_openai import OpenAIEmbeddings

from models.candidate import CandidateProfile

logger = logging.getLogger(__name__)


_model = None


def build_candidate_text(c: CandidateProfile) -> str:
    """
    Combine all candidate fields into one string for embedding.
    """
    parts = []
    if c.top_skills:
        parts.append(f"Expert skills: {c.top_skills}")
    if c.industry:
        parts.append(f"Industry: {c.industry}")
    if c.years_of_experience:
        parts.append(f"{c.years_of_experience} years of experience")
    if c.name:
        parts.append(c.name)
    if c.current_title and c.current_company:
        parts.append(f"{c.current_title} at {c.current_company}")
    if c.city or c.country:
        parts.append(f"Location: {', '.join(filter(None, [c.city, c.country]))}")
    if c.headline:
        parts.append(c.headline)
    if c.skills:
        parts.append(f"Skills: {c.skills}")
    if c.languages:
        parts.append(f"Languages: {c.languages}")
    if c.education:
        parts.append(f"Education: {c.education}")
    if c.work_history:
        parts.append(f"Work history: {c.work_history}")
    if c.job_description:
        parts.append(c.job_description[:300])

    return ". ".join(parts)


def get_embedding_model():
        logger.info("Using OpenRouter embedding model: text-embedding-ada-002")
        try:
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        except Exception as e:
            logger.error("Failed to initialise OpenRouter embedding client: %s", e)
            raise




def embed_texts(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        _model = get_embedding_model()
    try:
        logger.debug("Embedding %d texts", len(texts))
        vectors = _model.embed_documents(texts)
        logger.debug("Embedding complete | vectors=%d dims=%d", len(vectors), len(vectors[0]) if vectors else 0)
        return vectors
    except Exception as e:
        logger.error("embed_documents failed: %s", e)
        raise


def embed_query(text: str) -> list[float]:
    global _model
    if _model is None:
        _model = get_embedding_model()
    try:
        logger.debug("Embedding query: '%s'", text[:80])
        vector = _model.embed_query(text)
        logger.debug("Query embedding complete | dims=%d", len(vector))
        return vector
    except Exception as e:
        logger.error("embed_query failed: %s", e)
        raise
