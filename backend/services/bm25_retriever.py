from datasets import load_dataset
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, max_evidence=50000):

        self.max_evidence = max_evidence

        print("Loading FEVER dataset...")

        dataset = load_dataset(
            "copenlu/fever_gold_evidence"
        )

        data = dataset["train"].select(
            range(
                min(
                    self.max_evidence,
                    len(dataset["train"])
                )
            )
        )

        evidence_dict = {}

        for row in data:

            for item in row["evidence"]:

                if len(item) < 3:
                    continue

                title = item[0]
                sentence_id = item[1]
                text = item[2]

                key = (
                    title,
                    sentence_id,
                    text
                )

                evidence_dict[key] = {
                    "title": title,
                    "sentence_id": sentence_id,
                    "text": text
                }

        self.evidence = list(
            evidence_dict.values()
        )

        print(
            f"Unique evidence records: "
            f"{len(self.evidence)}"
        )

        texts = [
            item["text"]
            for item in self.evidence
        ]

        print("Creating BM25 index...")

        tokenized_texts = [
            text.lower().split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_texts
        )

        print("BM25 index ready.")

    def retrieve(self, claim, top_k=5):

        tokenized_query = claim.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for index in top_indices:

            evidence = self.evidence[index].copy()

            evidence["score"] = float(
                scores[index]
            )

            results.append(evidence)

        return results