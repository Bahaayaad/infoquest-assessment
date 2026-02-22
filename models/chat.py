from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from models.candidate import CandidateResult


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5


class ChatResponse(BaseModel):
    conversation_id: str
    query: str
    candidates: list[CandidateResult]
    summary: str
    timestamp: datetime = datetime.utcnow()

