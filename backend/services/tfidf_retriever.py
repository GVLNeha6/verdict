from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:

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

        print("Creating TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        self.matrix = self.vectorizer.fit_transform(
            texts
        )

        print("TF-IDF index ready.")

    def retrieve(self, claim, top_k=5):

        query_vector = self.vectorizer.transform(
            [claim]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            evidence = self.evidence[index].copy()

            evidence["score"] = float(
                scores[index]
            )

            results.append(evidence)

        return results