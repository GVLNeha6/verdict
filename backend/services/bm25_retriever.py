"""Lexical baseline: BM25 (Okapi) with the same tokenisation as TF-IDF."""
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from services.corpus import index_text, load_fever_evidence, tokenize


class BM25Retriever:
    def __init__(self, max_claims: int = 50000, include_title: bool = False,
                 evidence: Optional[List[Dict]] = None):
        self.evidence = evidence if evidence is not None else load_fever_evidence(max_claims)
        self.bm25 = BM25Okapi([tokenize(index_text(e, include_title)) for e in self.evidence])

    def retrieve(self, claim: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        scores = np.asarray(self.bm25.get_scores(tokenize(claim)))
        results = []
        for idx in np.argsort(-scores, kind="stable")[:top_k]:
            if scores[idx] <= 0 or scores[idx] < min_score:
                continue
            item = dict(self.evidence[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results
