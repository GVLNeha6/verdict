import os
import re

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from models.schemas import VerifyRequest, VerifyResponse
from services.evidence_retriever import FeverEvidenceRetriever
from services.debate_service import DebateService
from services.judge_service import JudgeService


load_dotenv()


app = FastAPI(
    title="Verdict API",
    description="Evidence-Based AI Claim Verification Platform",
    version="1.0.0"
)


# --------------------------------------------------
# Initialize services
# --------------------------------------------------

retriever = FeverEvidenceRetriever(
    max_evidence=50000
)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


debate_service = DebateService(
    api_key=api_key
)

judge_service = JudgeService(
    api_key=api_key
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def extract_verdict(text):
    """
    Extract SUPPORTS, REFUTES, or NOT ENOUGH INFO
    from the judge response.
    """

    match = re.search(
        r"VERDICT:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)",
        text.upper()
    )

    if match:
        return match.group(1)

    return "NOT ENOUGH INFO"


def extract_confidence(text):
    """
    Extract confidence value from the judge response.
    """

    match = re.search(
        r"CONFIDENCE:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        text.upper()
    )

    if match:
        return float(match.group(1))

    return 0.0


def extract_explanation(text):
    """
    Extract explanation from the judge response.
    """

    match = re.search(
        r"EXPLANATION:\s*(.*?)(?:\nEVIDENCE_USED:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return text.strip()


# --------------------------------------------------
# Basic endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Verdict API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# Claim verification endpoint
# --------------------------------------------------

@app.post(
    "/api/verify",
    response_model=VerifyResponse
)
def verify_claim(request: VerifyRequest):

    claim = request.claim.strip()

    if not claim:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty."
        )

    # --------------------------------------------------
    # 1. Retrieve evidence
    # --------------------------------------------------

    evidence = retriever.retrieve(
        claim,
        top_k=5
    )

    # --------------------------------------------------
    # 2. Run multi-agent debate
    # --------------------------------------------------

    debate_history = debate_service.run_debate(
        claim=claim,
        evidence=evidence,
        rounds=3
    )

    # --------------------------------------------------
    # 3. Judge the debate
    # --------------------------------------------------

    judge_result = judge_service.judge(
        claim=claim,
        evidence=evidence,
        debate_history=debate_history
    )

    # JudgeService currently returns a plain string
    raw_text = str(judge_result)

    # --------------------------------------------------
    # 4. Extract judge results
    # --------------------------------------------------

    verdict = extract_verdict(
        raw_text
    )

    confidence = extract_confidence(
        raw_text
    )

    explanation = extract_explanation(
        raw_text
    )

    # --------------------------------------------------
    # 5. Return final API response
    # --------------------------------------------------

    return {
        "claim": claim,

        "verdict": verdict,

        "confidence": confidence,

        "explanation": explanation,

        "evidence": evidence,

        "debate": {
            "agents": 2,
            "rounds": 3,
            "final_reasoning": debate_history[-1]
        }
    }
