from datetime import datetime
from typing import List

from pydantic import BaseModel

from models.candidate import CandidateResult


class ResearchRequest (BaseModel):
    query: str
    max_iterations: int = 5


class IterationLog(BaseModel):
    iteration: int
    thought: str
    action: str
    action_input: str
    reasoning: str
    observation: str | None = None

class ResearchResponse(BaseModel):
    query: str
    candidates: list[CandidateResult]
    total_found: int
    trace: List[IterationLog]
    timestamp: datetime

