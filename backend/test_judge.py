import os

from services.evidence_retriever import FeverEvidenceRetriever
from services.debate_service import DebateService
from services.judge_service import JudgeService


claim = (
    "The number of new cases of shingles per year "
    "extends from 1.2–3.4 per 1,000 among healthy individuals."
)


# --------------------------------------------------
# 1. Retrieve evidence
# --------------------------------------------------

print("Loading evidence retriever...")

retriever = FeverEvidenceRetriever(
    max_evidence=50000
)

evidence = retriever.retrieve(
    claim,
    top_k=5
)


print("\nRetrieved evidence:")

for i, item in enumerate(evidence, start=1):

    print(
        f"{i}. [{item['score']:.4f}] "
        f"{item['title']} - {item['text']}"
    )


# --------------------------------------------------
# 2. Run multi-agent debate
# --------------------------------------------------

print("\nStarting debate...")

debate_service = DebateService(
    api_key=os.environ["GEMINI_API_KEY"]
)

debate = debate_service.run_debate(
    claim=claim,
    evidence=evidence,
    rounds=3
)


print("\nDebate completed.")


# --------------------------------------------------
# 3. Run Judge
# --------------------------------------------------

print("\nStarting Judge...")

judge_service = JudgeService(
    api_key=os.environ["GEMINI_API_KEY"]
)

verdict = judge_service.judge(
    claim=claim,
    evidence=evidence,
    debate_history=debate
)


# --------------------------------------------------
# 4. Display final result
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)

print(verdict)