from services.tfidf_retriever import TfidfRetriever


retriever = TfidfRetriever(
    max_evidence=50000
)

claim = (
    "The number of new cases of shingles per year "
    "extends from 1.2–3.4 per 1,000 among healthy individuals."
)

results = retriever.retrieve(
    claim,
    top_k=5
)

print("\nCLAIM:")
print(claim)

print("\nTF-IDF RETRIEVED EVIDENCE:")

for i, result in enumerate(results, start=1):

    print(f"\n{i}. Score: {result['score']:.4f}")
    print(f"Title: {result['title']}")
    print(f"Sentence ID: {result['sentence_id']}")
    print(f"Text: {result['text']}")