"""Compare TF-IDF, BM25 and SBERT+FAISS on the SAME corpus with the SAME tokenisation.

Usage: python compare_retrieval.py --num-claims 1000 [--include-title]
Writes retrieval_comparison_results.json.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-claims", type=int, default=1000)
    ap.add_argument("--max-claims", type=int, default=50000, help="FEVER train claims indexed")
    ap.add_argument("--include-title", action="store_true")
    ap.add_argument("--output", default="retrieval_comparison_results.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from services.bm25_retriever import BM25Retriever
    from services.corpus import load_fever_evidence
    from services.evidence_retriever import FeverEvidenceRetriever
    from services.retrieval_eval import evaluate_retriever
    from services.tfidf_retriever import TfidfRetriever

    claims = load_dataset("copenlu/fever_gold_evidence", split="validation") \
        .select(range(args.num_claims))
    evidence = load_fever_evidence(args.max_claims)     # loaded once, shared by all three
    print(f"Corpus: {len(evidence)} unique evidence records")

    retrievers = {
        "TF-IDF + cosine": TfidfRetriever(evidence=evidence, include_title=args.include_title),
        "BM25": BM25Retriever(evidence=evidence, include_title=args.include_title),
        "Sentence-BERT + FAISS": FeverEvidenceRetriever(
            max_claims=args.max_claims, include_title=args.include_title, evidence=evidence),
    }
    results = {"config": vars(args), "corpus_size": len(evidence), "retrievers": {}}
    for name, r in retrievers.items():
        results["retrievers"][name] = evaluate_retriever(r, claims)

    print(f"\n{'Algorithm':<24}{'n':>5}{'hit@1':>9}{'hit@3':>9}{'hit@5':>9}")
    for name, m in results["retrievers"].items():
        print(f"{name:<24}{m['claims_evaluated']:>5}" +
              "".join(f"{m[f'hit@{k}']*100:>8.1f}%" for k in (1, 3, 5)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
