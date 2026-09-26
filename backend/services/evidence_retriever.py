from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class FeverEvidenceRetriever:

    def __init__(self, max_evidence=50000):

        self.max_evidence = max_evidence

        self.index_dir = (
            Path(__file__).resolve().parent.parent
            / "retriever_index"
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index_path = (
            self.index_dir / "fever.index"
        )

        self.metadata_path = (
            self.index_dir / "evidence.npy"
        )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        if (
            self.index_path.exists()
            and self.metadata_path.exists()
        ):

            print("Loading existing FEVER index...")

            self.index = faiss.read_index(
                str(self.index_path)
            )

            self.evidence = np.load(
                self.metadata_path,
                allow_pickle=True
            ).tolist()

            print(
                f"Loaded {len(self.evidence)} "
                f"evidence records."
            )

        else:

            print("No saved index found.")
            print("Building FEVER evidence index...")

            self._build_index()

    def _build_index(self):

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

        print("Creating embeddings...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype("float32")

        faiss.normalize_L2(
            embeddings
        )

        print("Creating FAISS index...")

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(
            embeddings
        )

        print("Saving FAISS index...")

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        np.save(
            self.metadata_path,
            np.array(
                self.evidence,
                dtype=object
            )
        )

        print("FEVER evidence index saved.")

    def retrieve(
        self,
        claim,
        top_k=5
    ):

        query_embedding = self.model.encode(
            [claim],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            evidence = (
                self.evidence[index].copy()
            )

            evidence["score"] = float(
                score
            )

            results.append(evidence)

        return results