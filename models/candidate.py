from pydantic import BaseModel
from typing import Optional

class CandidateProfile(BaseModel):
    id: str
    name: str
    headline: Optional[str] = None
    email: Optional[str] = None
    years_of_experience: Optional[int] = None
    city: Optional[str] = None
    country: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    industry: Optional[str] = None
    skills: Optional[str] = None
    top_skills: Optional[str] = None
    education: Optional[str] = None
    languages: Optional[str] = None
    work_history: Optional[str] = None
    job_description: Optional[str] = None


class CandidateResult(BaseModel):
    id: str
    name: str
    headline: Optional[str] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    years_of_experience: Optional[int] = None
    skills: Optional[str] = None
    languages: Optional[str] = None
    education: Optional[str] = None
    relevance_score: float
    why_match: str
    highlights: list[str]
