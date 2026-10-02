"""Retriever tests with a fake embedding model (no network / model download)."""
import hashlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

CORPUS = [
    {"title": "Saxony", "sentence_id": "0", "text": "Saxony is a German state in the east."},
    {"title": "Paris", "sentence_id": "1", "text": "Paris is the capital of France."},
    {"title": "Tokyo", "sentence_id": "2", "text": "Tokyo is the capital of Japan."},
    {"title": "Cats", "sentence_id": "3", "text": "Cats are small carnivorous mammals."},
]


class FakeST:
    encodes = 0

    def __init__(self, name):
        pass

    def encode(self, texts, **kw):
        FakeST.encodes += 1
        out = np.zeros((len(texts), 64), dtype="float32")
        for i, t in enumerate(texts):                    # bag-of-words hashing embedding
            for w in t.lower().replace(".", "").split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return out


@pytest.fixture(autouse=True)
def fake_sbert(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    sys.modules.pop("services.evidence_retriever", None)
    FakeST.encodes = 0


def test_dense_retrieval_cache_and_invalidation(tmp_path):
    from services.evidence_retriever import FeverEvidenceRetriever as R

    r = R(max_claims=10, evidence=CORPUS, index_dir=tmp_path)
    top = r.retrieve("capital of France", top_k=2)
    assert top[0]["title"] == "Paris" and top[0]["score"] > 0
    assert (tmp_path / "evidence.json").exists() and not list(tmp_path.glob("*.npy"))

    before = FakeST.encodes
    R(max_claims=10, evidence=CORPUS, index_dir=tmp_path)            # identical -> reuse
    assert FakeST.encodes == before
    r2 = R(max_claims=10, evidence=CORPUS, index_dir=tmp_path)       # sanity
    assert len(r2.evidence) == 4

    R(max_claims=10, include_title=True, evidence=CORPUS, index_dir=tmp_path)   # config change
    assert FakeST.encodes > before                                    # -> rebuilt, not stale

    R(max_claims=10, evidence=CORPUS[:3], index_dir=tmp_path)         # different corpus
    import json
    assert json.loads((tmp_path / "meta.json").read_text())["n_records"] == 3


def test_min_score_filters(tmp_path):
    from services.evidence_retriever import FeverEvidenceRetriever as R
    r = R(max_claims=10, evidence=CORPUS, index_dir=tmp_path)
    assert r.retrieve("capital of France", top_k=4, min_score=0.99) == []


def test_lexical_retrievers_share_tokenisation_and_drop_zero_scores():
    from services.bm25_retriever import BM25Retriever
    from services.tfidf_retriever import TfidfRetriever

    for cls in (BM25Retriever, TfidfRetriever):
        r = cls(evidence=CORPUS)
        res = r.retrieve("The capital of Japan", top_k=4)
        assert res[0]["title"] == "Tokyo"
        assert all(x["score"] > 0 for x in res)
        assert r.retrieve("zzzz qqqq", top_k=4) == []
