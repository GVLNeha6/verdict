'''import os

from dotenv import load_dotenv

from services.evidence_retriever import FeverEvidenceRetriever
from services.debate_service import DebateService


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


retriever = FeverEvidenceRetriever(
    max_evidence=50000
)

debate_service = DebateService(
    api_key=api_key
)


claim = (
    "The number of new cases of shingles per year "
    "extends from 1.2–3.4 per 1,000 among healthy individuals."
)


print("=" * 70)
print("TESTING EVIDENCE + DEBATE SERVICE")
print("=" * 70)


print("\nRetrieving evidence...\n")

evidence = retriever.retrieve(
    claim,
    top_k=5
)


print("RETRIEVED EVIDENCE")
print("=" * 70)

for i, item in enumerate(evidence):

    print(f"\nEvidence {i + 1}")
    print(f"Title: {item['title']}")
    print(f"Sentence ID: {item['sentence_id']}")
    print(f"Score: {item['score']:.4f}")
    print(f"Text: {item['text']}")


print("\n" + "=" * 70)
print("RUNNING 2 AGENTS × 3 ROUNDS")
print("=" * 70)


results = debate_service.run_debate(
    claim=claim,
    evidence=evidence,
    rounds=3
)


for result in results:

    print("\n" + "=" * 70)
    print(f"ROUND {result['round']}")
    print("=" * 70)

    print("\nAGENT 1:")
    print(result["agent_1"])

    print("\nAGENT 2:")
    print(result["agent_2"])


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
'''