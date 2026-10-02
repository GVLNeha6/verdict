"""Lexical baseline: TF-IDF + cosine similarity (same tokenisation as BM25)."""
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.corpus import index_text, load_fever_evidence, tokenize


class TfidfRetriever:
    def __init__(self, max_claims: int = 50000, include_title: bool = False,
                 evidence: Optional[List[Dict]] = None):
        self.evidence = evidence if evidence is not None else load_fever_evidence(max_claims)
        texts = [index_text(e, include_title) for e in self.evidence]
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False,
                                          token_pattern=None)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, claim: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        scores = cosine_similarity(self.vectorizer.transform([claim]), self.matrix)[0]
        results = []
        for idx in np.argsort(-scores, kind="stable")[:top_k]:
            if scores[idx] <= 0 or scores[idx] < min_score:
                continue
            item = dict(self.evidence[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results
