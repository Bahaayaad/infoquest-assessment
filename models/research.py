from datetime import datetime
from typing import List

from pydantic import BaseModel, field_validator

from models.candidate import CandidateResult


class ResearchRequest (BaseModel):
    query: str
    max_iterations: int = 4
    min_results: int = 5

    @field_validator('max_iterations')
    @classmethod
    def cap_max_iterations(cls, v: int) -> int:
        if v > 5:
            return 5
        return v

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
    iterations_ran: int
    stop_reason: str
    react_trace: list[IterationLog]
    timestamp: datetime = datetime.utcnow()
