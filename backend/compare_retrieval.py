from datasets import load_dataset

from services.tfidf_retriever import TfidfRetriever
from services.bm25_retriever import BM25Retriever
from services.evidence_retriever import FeverEvidenceRetriever


NUM_CLAIMS = 1000
TOP_K_VALUES = [1, 3, 5]


def get_gold_ids(row):

    gold_ids = set()

    for item in row["evidence"]:

        if len(item) >= 2:

            title = item[0]
            sentence_id = item[1]

            gold_ids.add(
                (
                    title,
                    sentence_id
                )
            )

    return gold_ids


def evaluate_retriever(
    name,
    retriever,
    claims
):

    print(
        f"\nEvaluating {name}..."
    )

    covered_claims = 0

    recall_counts = {
        k: 0
        for k in TOP_K_VALUES
    }

    evaluated_claims = 0

    corpus_ids = {
        (
            item["title"],
            str(item["sentence_id"])
        )
        for item in retriever.evidence
    }

    for row_number, row in enumerate(
        claims,
        start=1
    ):

        gold_ids = get_gold_ids(row)

        if not gold_ids:
            continue

        # Check whether at least one gold
        # evidence record exists in our corpus.
        if not gold_ids.intersection(
            corpus_ids
        ):
            continue

        covered_claims += 1

        results = retriever.retrieve(
            row["claim"],
            top_k=max(TOP_K_VALUES)
        )

        retrieved_ids = [
            (
                item["title"],
                str(item["sentence_id"])
            )
            for item in results
        ]

        evaluated_claims += 1

        for k in TOP_K_VALUES:

            top_k_ids = set(
                retrieved_ids[:k]
            )

            if gold_ids.intersection(
                top_k_ids
            ):

                recall_counts[k] += 1

        if row_number % 20 == 0:

            print(
                f"Processed "
                f"{row_number}/{len(claims)} claims"
            )

    print(
        f"\n{name}"
    )

    print(
        f"Gold-evidence-covered claims: "
        f"{covered_claims}/{len(claims)}"
    )

    print(
        f"Evaluated claims: "
        f"{evaluated_claims}"
    )

    results = {
        "algorithm": name,
        "covered": covered_claims,
        "evaluated": evaluated_claims
    }

    for k in TOP_K_VALUES:

        if evaluated_claims == 0:

            recall = 0

        else:

            recall = (
                recall_counts[k]
                / evaluated_claims
                * 100
            )

        results[
            f"recall@{k}"
        ] = recall

        print(
            f"Recall@{k}: "
            f"{recall:.2f}%"
        )

    return results


print(
    "Loading FEVER validation set..."
)

dataset = load_dataset(
    "copenlu/fever_gold_evidence"
)

claims = dataset["validation"].select(
    range(NUM_CLAIMS)
)

print(
    f"Using {len(claims)} validation claims."
)


print(
    "\nInitializing TF-IDF..."
)

tfidf = TfidfRetriever(
    max_evidence=50000
)


print(
    "\nInitializing BM25..."
)

bm25 = BM25Retriever(
    max_evidence=50000
)


print(
    "\nInitializing Sentence-BERT + FAISS..."
)

sbert = FeverEvidenceRetriever(
    max_evidence=50000
)


results = []


results.append(
    evaluate_retriever(
        "TF-IDF + Cosine",
        tfidf,
        claims
    )
)


results.append(
    evaluate_retriever(
        "BM25",
        bm25,
        claims
    )
)


results.append(
    evaluate_retriever(
        "Sentence-BERT + FAISS",
        sbert,
        claims
    )
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL RETRIEVAL COMPARISON"
)

print(
    "=" * 70
)

print(
    f"{'Algorithm':<25}"
    f"{'Coverage':<12}"
    f"{'Recall@1':<12}"
    f"{'Recall@3':<12}"
    f"{'Recall@5':<12}"
)

print(
    "-" * 70
)


for result in results:

    print(
        f"{result['algorithm']:<25}"
        f"{result['covered']:<12}"
        f"{result['recall@1']:<12.2f}"
        f"{result['recall@3']:<12.2f}"
        f"{result['recall@5']:<12.2f}"
    )