import os

from services.evidence_retriever import FeverEvidenceRetriever
from services.debate_service import DebateService


claim = (
    "The number of new cases of shingles per year "
    "extends from 1.2–3.4 per 1,000 among healthy individuals."
)


print("Loading evidence retriever...")

retriever = FeverEvidenceRetriever(
    max_evidence=50000
)

evidence = retriever.retrieve(
    claim,
    top_k=5
)


print("\nStarting debate...")

debate_service = DebateService(
    api_key=os.environ["GEMINI_API_KEY"]
)

debate = debate_service.run_debate(
    claim=claim,
    evidence=evidence,
    rounds=3
)


for round_data in debate:

    print("\n" + "=" * 60)
    print(f"ROUND {round_data['round']}")
    print("=" * 60)

    print("\nAGENT 1:")
    print(round_data["agent_1"])

    print("\nAGENT 2:")
    print(round_data["agent_2"])