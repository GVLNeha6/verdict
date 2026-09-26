from typing import List
from pydantic import BaseModel


class VerifyRequest(BaseModel):
    claim: str


class EvidenceItem(BaseModel):
    title: str
    sentence_id: str
    text: str
    score: float


class DebateResult(BaseModel):
    agents: int
    rounds: int
    final_reasoning: dict


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence: List[EvidenceItem]
    debate: DebateResult