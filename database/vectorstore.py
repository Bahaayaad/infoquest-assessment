"""
Thin wrapper around ChromaDB.
Stores candidate vectors + metadata, and searches them.
"""

import chromadb

from models.candidate import CandidateProfile

COLLECTION_NAME = "candidates"

# One persistent client for the whole app
_client = chromadb.PersistentClient(path="../chroma_db")
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def upsert_candidates(candidates: list[CandidateProfile], embeddings: list[list[float]]):
    _collection.upsert(
        ids=[c.id for c in candidates],
        embeddings=embeddings,
        metadatas=[_build_metadata(c) for c in candidates],
        documents=[c.headline or c.name for c in candidates],
    )


def search(query_vector: list[float], top_k: int = 5 , where: dict = None) -> list[dict]:
    if _collection.count() == 0:
        return []

    kwargs = {
        "query_embeddings": [query_vector],
        "n_results": min(top_k, _collection.count()),
        "include": ["metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = _collection.query(**kwargs)

    matches = []
    for cid, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - (distance / 2), 4)  # cosine distance → similarity
        matches.append({"id": cid, "score": similarity, **metadata})


    return matches


def count() -> int:
    return _collection.count()


def wipe():
    _client.delete_collection(COLLECTION_NAME)
    # Recreate so the app can keep using _collection reference
    global _collection
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _build_metadata(c: CandidateProfile) -> dict:
    return {
        "name":               c.name or "",
        "headline":           c.headline or "",
        "current_title":      c.current_title or "",
        "current_company":    c.current_company or "",
        "industry":           c.industry or "",
        "city":               c.city or "",
        "country":            c.country or "",
        "years_of_experience": c.years_of_experience or 0,
        "skills":             c.skills or "",
        "top_skills":         c.top_skills or "",
        "education":          (c.education or "")[:400],
        "languages":          c.languages or "",
        "email":              c.email or "",
    }
