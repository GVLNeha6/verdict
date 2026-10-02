from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from config import MAX_CLAIM_CHARS

Verdict = Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=MAX_CLAIM_CHARS)


class EvidenceItem(BaseModel):
    title: str
    sentence_id: str
    text: str
    score: float


class DebateResult(BaseModel):
    agents: int
    rounds: int
    final_reasoning: Dict[str, str]


class VerifyResponse(BaseModel):
    claim: str
    verdict: Verdict
    confidence: Optional[float] = None
    explanation: str
    evidence: List[EvidenceItem]            # everything retrieved
    evidence_used: List[int]                # 1-based indices the judge cited (validated)
    grounded: bool                          # False if SUPPORTS/REFUTES cites no valid evidence
    debate: DebateResult
