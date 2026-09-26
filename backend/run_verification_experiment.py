import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset

from services.evidence_retriever import FeverEvidenceRetriever
from services.debate_service import DebateService
from services.judge_service import JudgeService


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_CLAIMS = 10
TOP_K = 5
ROUNDS = 3

MODEL = "gemini-3.5-flash-lite"

BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai"
)

OUTPUT_FILE = Path(
    "verification_experiment_results.json"
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def extract_verdict(text):
    """
    Extract FEVER-style verdict from an LLM response.
    """

    match = re.search(
        r"VERDICT:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)",
        text.upper()
    )

    if match:
        return match.group(1)

    return "NOT ENOUGH INFO"


def save_results(results):
    """
    Save results after every completed claim.
    """

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )


def load_previous_results():
    """
    Resume an interrupted experiment if possible.
    """

    if not OUTPUT_FILE.exists():
        return []

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def get_gold_ids(row):
    """
    Extract FEVER gold evidence identifiers.
    """

    gold_ids = set()

    for item in row["evidence"]:

        if len(item) >= 2:

            gold_ids.add(
                (item[0], item[1])
            )

    return gold_ids


def find_evidence_covered_claims(
    dataset,
    retriever,
    num_claims
):
    """
    Select validation claims whose gold evidence
    exists in the current retrieval corpus.

    The current corpus contains evidence extracted
    from the first 50,000 FEVER training claims.
    """

    corpus_ids = set()

    for item in retriever.evidence:

        corpus_ids.add(
            (
                item["title"],
                item["sentence_id"]
            )
        )

    selected = []

    for row in dataset:

        gold_ids = get_gold_ids(row)

        if not gold_ids:
            continue

        if gold_ids.intersection(corpus_ids):

            selected.append(row)

        if len(selected) >= num_claims:
            break

    return selected


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("FINAL FEVER VERIFICATION EXPERIMENT")
    print("=" * 70)

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading FEVER validation dataset...")

    dataset = load_dataset(
        "copenlu/fever_gold_evidence",
        split="validation"
    )

    print(
        f"Validation claims available: {len(dataset)}"
    )

    # --------------------------------------------------
    # Load retriever
    # --------------------------------------------------

    print("\nLoading evidence retriever...")

    retriever = FeverEvidenceRetriever(
        max_evidence=50000
    )

    print(
        f"Retriever evidence records: "
        f"{len(retriever.evidence)}"
    )

    # --------------------------------------------------
    # Select claims
    # --------------------------------------------------

    print(
        f"\nSelecting {NUM_CLAIMS} claims "
        "with evidence available in the corpus..."
    )

    claims = find_evidence_covered_claims(
        dataset,
        retriever,
        NUM_CLAIMS
    )

    if len(claims) < NUM_CLAIMS:

        raise RuntimeError(
            f"Only {len(claims)} suitable claims found."
        )

    print(
        f"Selected {len(claims)} claims."
    )

    # --------------------------------------------------
    # Initialize services
    # --------------------------------------------------

    debate_service = DebateService(
        api_key=api_key,
        model=MODEL,
        base_url=BASE_URL,
        agents=2
    )

    judge_service = JudgeService(
        api_key=api_key,
        model=MODEL,
        base_url=BASE_URL
    )

    # --------------------------------------------------
    # Resume previous results
    # --------------------------------------------------

    results = load_previous_results()

    completed_claim_ids = {
        item["id"]
        for item in results
        if "id" in item
    }

    if results:

        print(
            f"\nResuming experiment. "
            f"Already completed: {len(results)}"
        )

    # --------------------------------------------------
    # Process claims
    # --------------------------------------------------

    for index, row in enumerate(
        claims,
        start=1
    ):

        claim_id = str(row["id"])

        if claim_id in completed_claim_ids:

            print(
                f"\nSkipping claim {index}: "
                "already completed."
            )

            continue

        claim = row["claim"]
        gold_label = row["label"]

        print("\n" + "=" * 70)
        print(
            f"CLAIM {index}/{len(claims)}"
        )
        print("=" * 70)

        print(
            f"Gold label: {gold_label}"
        )

        print(
            f"Claim: {claim}"
        )

        result = {
            "id": claim_id,
            "claim": claim,
            "gold_label": gold_label
        }

        # --------------------------------------------------
        # A. Single LLM baseline
        # --------------------------------------------------

        print("\n[A] Single LLM baseline...")

        single_prompt = f"""
You are a claim verification system.

Determine whether the following claim is:

SUPPORTS
REFUTES
NOT ENOUGH INFO

Claim:
{claim}

Use your own reasoning.

Return exactly:

VERDICT: <SUPPORTS / REFUTES / NOT ENOUGH INFO>
"""

        single_response = debate_service._ask(
            single_prompt
        )

        single_verdict = extract_verdict(
            single_response
        )

        result["single_llm"] = {
            "verdict": single_verdict,
            "raw_response": single_response
        }

        print(
            f"Single LLM verdict: {single_verdict}"
        )

        # --------------------------------------------------
        # B. Debate without evidence
        # --------------------------------------------------

        print("\n[B] Debate without evidence...")

        debate_only_history = (
            debate_service.run_debate(
                claim=claim,
                evidence=None,
                rounds=ROUNDS
            )
        )

        debate_only_judge_response = (
            judge_service.judge(
                claim=claim,
                evidence=[],
                debate_history=debate_only_history
            )
        )

        debate_only_verdict = extract_verdict(
            debate_only_judge_response
        )

        result["debate_only"] = {
            "verdict": debate_only_verdict,
            "debate_history": debate_only_history,
            "judge_response": debate_only_judge_response
        }

        print(
            f"Debate-only verdict: "
            f"{debate_only_verdict}"
        )

        # --------------------------------------------------
        # C. Evidence + Debate
        # --------------------------------------------------

        print("\n[C] Evidence + Debate...")

        evidence = retriever.retrieve(
            claim,
            top_k=TOP_K
        )

        print(
            f"Retrieved {len(evidence)} evidence items."
        )

        evidence_debate_history = (
            debate_service.run_debate(
                claim=claim,
                evidence=evidence,
                rounds=ROUNDS
            )
        )

        evidence_judge_response = (
            judge_service.judge(
                claim=claim,
                evidence=evidence,
                debate_history=evidence_debate_history
            )
        )

        evidence_verdict = extract_verdict(
            evidence_judge_response
        )

        result["evidence_debate"] = {
            "verdict": evidence_verdict,
            "evidence": evidence,
            "debate_history": evidence_debate_history,
            "judge_response": evidence_judge_response
        }

        print(
            f"Evidence + Debate verdict: "
            f"{evidence_verdict}"
        )

        # --------------------------------------------------
        # Accuracy for this claim
        # --------------------------------------------------

        result["correct"] = {
            "single_llm": (
                single_verdict == gold_label
            ),
            "debate_only": (
                debate_only_verdict == gold_label
            ),
            "evidence_debate": (
                evidence_verdict == gold_label
            )
        }

        results.append(result)

        # Save immediately
        save_results(results)

        print("\nSaved progress.")

        print(
            f"Single LLM correct: "
            f"{result['correct']['single_llm']}"
        )

        print(
            f"Debate correct: "
            f"{result['correct']['debate_only']}"
        )

        print(
            f"Evidence + Debate correct: "
            f"{result['correct']['evidence_debate']}"
        )

        # Small pause between claims
        time.sleep(5)

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    total = len(results)

    if total == 0:

        print("No completed claims.")

        return

    single_correct = sum(
        item["correct"]["single_llm"]
        for item in results
    )

    debate_correct = sum(
        item["correct"]["debate_only"]
        for item in results
    )

    evidence_correct = sum(
        item["correct"]["evidence_debate"]
        for item in results
    )

    print(
        f"\nClaims evaluated: {total}"
    )

    print(
        f"\nSingle LLM:"
        f" {single_correct}/{total}"
        f" = {single_correct / total * 100:.2f}%"
    )

    print(
        f"Debate only:"
        f" {debate_correct}/{total}"
        f" = {debate_correct / total * 100:.2f}%"
    )

    print(
        f"Evidence + Debate:"
        f" {evidence_correct}/{total}"
        f" = {evidence_correct / total * 100:.2f}%"
    )

    print(
        f"\nDetailed results saved to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()