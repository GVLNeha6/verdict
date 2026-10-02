"""Retrieval evaluation (no heavy dependencies, so it is unit-testable)."""
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from services.corpus import corpus_ids, get_gold_ids


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate_retriever(retriever, claims: Iterable, ks: Sequence[int] = (1, 3, 5)) -> Dict:
    """Evaluate on claims that have >=1 gold sentence inside the retriever's corpus.

    hit@k          : share of claims where ANY gold sentence is in the top k
                     (the metric previously labelled "Recall@k").
    gold_recall@k  : mean fraction of a claim's in-corpus gold sentences found in the top k.
    """
    in_corpus = corpus_ids(retriever.evidence)
    max_k = max(ks)
    total = examined = 0
    hits = {k: 0 for k in ks}
    frac = {k: 0.0 for k in ks}

    for row in claims:
        total += 1
        gold = get_gold_ids(row) & in_corpus
        if not gold:
            continue
        examined += 1
        retrieved = [(r["title"], str(r["sentence_id"]))
                     for r in retriever.retrieve(row["claim"], top_k=max_k)]
        for k in ks:
            found = gold & set(retrieved[:k])
            hits[k] += bool(found)
            frac[k] += len(found) / len(gold)

    out = {"claims_total": total, "claims_evaluated": examined}
    for k in ks:
        lo, hi = wilson_interval(hits[k], examined)
        out[f"hit@{k}"] = hits[k] / examined if examined else 0.0
        out[f"hit@{k}_ci95"] = [lo, hi]
        out[f"gold_recall@{k}"] = frac[k] / examined if examined else 0.0
    return out
