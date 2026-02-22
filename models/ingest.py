from pydantic import BaseModel

class IngestRequest(BaseModel):
    force_reingest: bool = False

class IngestResponse(BaseModel):
    status: str
    total_processed: int
    message: str