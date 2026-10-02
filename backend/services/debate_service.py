"""Multi-agent debate (Du et al., 2023) with optional retrieved evidence.

Mechanism
  * Every agent keeps its own private message history.
  * Round 1: each agent answers independently.
  * Round r > 1: each agent is shown the *previous round's* answer of every OTHER agent,
    then answers again. Its own history keeps growing (user prompt, assistant answer, ...).
  * The previous-round answers are snapshotted before any agent speaks in the current round,
    so the update is simultaneous: an agent can never see a same-round answer.
"""
import logging
from typing import Dict, List, Optional

from services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_VERDICT_HINT = ("End with a line of the form 'CURRENT VERDICT: SUPPORTS', "
                 "'CURRENT VERDICT: REFUTES' or 'CURRENT VERDICT: NOT ENOUGH INFO'.")


def format_evidence_items(evidence: List[Dict]) -> str:
    return "\n\n".join(
        f"Evidence {i + 1}:\nTitle: {e['title']}\nText: {e['text']}"
        for i, e in enumerate(evidence)
    )


def format_evidence(evidence: Optional[List[Dict]]) -> str:
    if not evidence:
        return ""
    return "\n\nRetrieved evidence:\n" + format_evidence_items(evidence)


class DebateService:
    def __init__(self, llm: LLMClient, agents: int = 2, temperature: float = 0.7):
        if agents < 2:
            raise ValueError("A debate needs at least 2 agents.")
        self.llm = llm
        self.agents = agents
        self.temperature = temperature

    # ------------------------------------------------------------------ prompts
    def _initial_prompt(self, claim: str, evidence_text: str) -> str:
        if evidence_text:
            guidance = ("Base your analysis on the retrieved evidence. If it does not "
                        "settle the claim, say so.")
        else:
            guidance = "Use your own knowledge and reasoning."
        return (
            "You are an independent agent in a multi-agent fact-checking debate.\n\n"
            f"Claim:\n{claim}\n{evidence_text}\n\n"
            f"Analyse the claim independently. {guidance}\n"
            f"Give your reasoning. {_VERDICT_HINT}"
        )

    def _debate_prompt(self, claim: str, evidence_text: str, round_number: int,
                       others: List[str]) -> str:
        return (
            f"This is round {round_number} of the debate.\n\n"
            f"Claim:\n{claim}\n{evidence_text}\n\n"
            "Previous-round responses from the other agents:\n"
            + "".join(others)
            + "\nUse them as additional information. Critically examine their reasoning "
              "and revise your answer if warranted.\n"
              f"Give your updated reasoning. {_VERDICT_HINT}"
        )

    # ------------------------------------------------------------------- debate
    def run_debate(self, claim: str, evidence: Optional[List[Dict]] = None,
                   rounds: int = 3) -> List[Dict]:
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        evidence_text = format_evidence(evidence)

        contexts = [
            [{"role": "user", "content": self._initial_prompt(claim, evidence_text)}]
            for _ in range(self.agents)
        ]
        history = []

        for round_index in range(rounds):
            # Snapshot of every agent's latest answer BEFORE this round starts.
            previous = ([ctx[-1]["content"] for ctx in contexts]
                        if round_index > 0 else None)
            round_results = {}

            for i, ctx in enumerate(contexts):
                if previous is not None:
                    others = [
                        f"\nAgent {j + 1}'s previous response:\n\n{previous[j]}\n"
                        for j in range(self.agents) if j != i
                    ]
                    ctx.append({"role": "user", "content": self._debate_prompt(
                        claim, evidence_text, round_index + 1, others)})

                answer = self.llm.chat(ctx, temperature=self.temperature)
                ctx.append({"role": "assistant", "content": answer})
                round_results[f"agent_{i + 1}"] = answer

            history.append({"round": round_index + 1, **round_results})
            logger.debug("debate round %d finished", round_index + 1)

        return history
