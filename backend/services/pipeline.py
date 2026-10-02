"""Claim -> evidence -> debate -> judge, independent of any web framework."""
from typing import Dict

from config import Settings
from services.debate_service import DebateService
from services.judge_service import JudgeService
from services.llm_client import LLMClient


class JudgeParseError(RuntimeError):
    """The judge answered, but not in a parseable format (even after retries)."""


class VerificationPipeline:
    def __init__(self, settings: Settings, retriever, debate: DebateService, judge: JudgeService):
        self.settings = settings
        self.retriever = retriever
        self.debate = debate
        self.judge = judge

    def verify(self, claim: str) -> Dict:
        s = self.settings
        evidence = self.retriever.retrieve(claim, top_k=s.top_k, min_score=s.min_evidence_score)
        history = self.debate.run_debate(claim, evidence=evidence, rounds=s.rounds)
        result = self.judge.judge(claim, evidence, history, use_evidence=True)
        if not result.parse_ok:
            raise JudgeParseError("Judge returned an unparseable verdict.")

        final = {k: v for k, v in history[-1].items() if k.startswith("agent_")}
        return {
            "claim": claim,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "evidence": [{**e, "sentence_id": str(e["sentence_id"])} for e in evidence],
            "evidence_used": result.evidence_used,
            "grounded": result.grounded,
            "debate": {"agents": s.agents, "rounds": s.rounds, "final_reasoning": final},
        }


def build_pipeline(settings: Settings) -> VerificationPipeline:
    from services.evidence_retriever import FeverEvidenceRetriever  # heavy imports, lazy

    llm = LLMClient(settings.llm_api_key, settings.model, settings.base_url,
                    timeout=settings.llm_timeout)
    retriever = FeverEvidenceRetriever(max_claims=settings.corpus_max_claims,
                                       include_title=settings.include_title)
    return VerificationPipeline(
        settings, retriever,
        DebateService(llm, agents=settings.agents, temperature=settings.agent_temperature),
        JudgeService(llm, temperature=settings.judge_temperature),
    )
