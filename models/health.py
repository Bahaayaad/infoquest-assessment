from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    candidates_in_db: int
    candidates_indexed: int
