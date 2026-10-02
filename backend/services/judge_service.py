"""Final judge. Also serves the no-debate baselines so that all experimental conditions
share exactly the same decision prompt and differ only in (evidence, debate)."""
import logging
import re
from typing import Dict, List, Optional

from services.debate_service import format_evidence_items
from services.llm_client import LLMClient
from services.verdict_service import JudgeResult, parse_judge_response

logger = logging.getLogger(__name__)

_FORMAT = """Return your answer in exactly this format:

VERDICT: <one of SUPPORTS, REFUTES, NOT ENOUGH INFO>
CONFIDENCE: <number between 0 and 1>
EXPLANATION: <short explanation>
EVIDENCE_USED: <evidence numbers you relied on, e.g. 1, 3; write None if not applicable>"""

_LABEL_DEFS = """Labels:
- SUPPORTS: the claim is true.
- REFUTES: the claim is false.
- NOT ENOUGH INFO: the claim cannot be decided."""


def _agent_keys(round_entry: Dict) -> List[str]:
    keys = [k for k in round_entry if re.fullmatch(r"agent_\d+", k)]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


class JudgeService:
    def __init__(self, llm: LLMClient, temperature: float = 0.0, parse_retries: int = 1):
        self.llm = llm
        self.temperature = temperature
        self.parse_retries = parse_retries

    def build_prompt(self, claim: str, evidence: List[Dict],
                     debate_history: Optional[List[Dict]], use_evidence: bool) -> str:
        parts = ["You are the final judge in a claim verification system.",
                 f"CLAIM:\n{claim}"]

        if use_evidence:
            parts.append("RETRIEVED EVIDENCE:\n" + format_evidence_items(evidence))
            parts.append(
                "Decide whether the claim is supported, refuted, or not decidable from "
                "the evidence.\n" + _LABEL_DEFS + "\n"
                "Rules:\n- Judge strictly against the supplied evidence.\n"
                "- Do not use outside knowledge and do not invent evidence.\n"
                "- Use NOT ENOUGH INFO when the evidence does not settle the claim.\n"
                "- Cite the evidence numbers you used.")
        else:
            parts.append(
                "No evidence is available. Decide using your own knowledge and reasoning.\n"
                + _LABEL_DEFS + "\n"
                "Rules:\n- Commit to SUPPORTS or REFUTES when you can determine the answer "
                "from your knowledge.\n"
                "- Use NOT ENOUGH INFO only if the claim genuinely cannot be determined.\n"
                "- Write EVIDENCE_USED: None.")

        if debate_history:
            final = debate_history[-1]
            for key in _agent_keys(final):
                n = key.split("_")[1]
                parts.append(f"FINAL AGENT {n} REASONING:\n{final[key]}")
            parts.append("Do not pick a verdict merely because the agents agree; "
                         "check it against the rules above.")

        parts.append(_FORMAT)
        return "\n\n".join(parts)

    def judge(self, claim: str, evidence: Optional[List[Dict]] = None,
              debate_history: Optional[List[Dict]] = None,
              use_evidence: bool = True) -> JudgeResult:
        evidence = evidence or []

        if use_evidence and not evidence:
            # Nothing was retrieved: deterministic NEI, no LLM call, cannot hallucinate.
            return JudgeResult(verdict="NOT ENOUGH INFO", confidence=1.0,
                               explanation="No evidence was retrieved for this claim.",
                               grounded=True, parse_ok=True, raw="")

        prompt = self.build_prompt(claim, evidence, debate_history, use_evidence)
        n_evidence = len(evidence) if use_evidence else 0

        result = None
        for attempt in range(1 + self.parse_retries):
            raw = self.llm.ask(prompt, temperature=self.temperature)
            result = parse_judge_response(raw, n_evidence)
            if result.parse_ok:
                break
            logger.warning("Judge output unparseable (attempt %d): %r", attempt + 1, raw[:200])

        if not use_evidence:
            result.grounded = result.parse_ok   # grounding is not applicable without evidence
        return result
