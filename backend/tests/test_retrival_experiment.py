import json

import pytest

from run_verification_experiment import load_or_init, save_atomic, select_claims
from services.corpus import clean_title, get_gold_ids, index_text, tokenize
from services.retrieval_eval import evaluate_retriever


class FakeRetriever:
    def __init__(self, ranked):
        self.evidence = [{"title": "A", "sentence_id": "0", "text": "x"},
                         {"title": "B", "sentence_id": "1", "text": "y"},
                         {"title": "C", "sentence_id": "2", "text": "z"}]
        self.ranked = ranked

    def retrieve(self, claim, top_k=5, min_score=0.0):
        return [self.evidence[i] | {"score": 1.0} for i in self.ranked[:top_k]]


def test_evaluate_retriever_hit_and_gold_recall():
    claims = [
        {"claim": "c1", "evidence": [["A", 0, "x"], ["B", 1, "y"]]},   # both gold in corpus
        {"claim": "c2", "evidence": [["Z", 9, "q"]]},                   # not in corpus -> skipped
    ]
    m = evaluate_retriever(FakeRetriever([2, 1, 0]), claims, ks=(1, 3))
    assert m["claims_total"] == 2 and m["claims_evaluated"] == 1
    assert m["hit@1"] == 0.0 and m["hit@3"] == 1.0
    assert m["gold_recall@3"] == 1.0
    m2 = evaluate_retriever(FakeRetriever([1, 2, 0]), claims, ks=(1, 3))
    assert m2["hit@1"] == 1.0 and m2["gold_recall@1"] == 0.5


def test_gold_ids_normalise_types():
    assert get_gold_ids({"evidence": [["T", 3, "s"], ["U"]]}) == {("T", "3")}


def test_tokenize_is_shared_and_removes_stopwords():
    assert tokenize("The Cat, and the Hat!") == ["cat", "hat"]
    assert clean_title("Diamonds_-LRB-Rihanna_song-RRB-") == "Diamonds (Rihanna song)"
    assert index_text({"title": "Saxony", "text": "It is a state."}, True) == "Saxony: It is a state."


def make_rows():
    rows = []
    for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
        for i in range(20):
            rows.append({"id": f"{label}{i}", "label": label, "claim": "c",
                         "evidence": [["T", i, "s"]]})
    return rows


def test_selection_is_stratified_seeded_and_covered_only():
    rows = make_rows()
    corpus = {("T", str(i)) for i in range(0, 20, 2)}         # only even ids covered (10/label)
    a = select_claims(rows, corpus, 9, seed=1)
    b = select_claims(rows, corpus, 9, seed=1)
    c = select_claims(rows, corpus, 9, seed=2)
    assert [r["id"] for r in a] == [r["id"] for r in b] != [r["id"] for r in c]
    assert {l: sum(r["label"] == l for r in a) for l in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")} \
        == {"SUPPORTS": 3, "REFUTES": 3, "NOT ENOUGH INFO": 3}
    with pytest.raises(RuntimeError):
        select_claims(rows, corpus, 45, seed=1)


CFG = {k: 1 for k in ("model", "rounds", "agents", "top_k", "seed", "num_claims",
                      "agent_temperature", "judge_temperature", "include_title", "corpus_max_claims")}


def test_resume_is_safe(tmp_path):
    p = tmp_path / "r.json"
    assert load_or_init(p, CFG, False)["results"] == []
    save_atomic(p, {"config": CFG, "results": [{"id": "1"}]})
    assert load_or_init(p, CFG, False)["results"] == [{"id": "1"}]
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(SystemExit):                      # config changed -> refuse to mix
        load_or_init(p, {**CFG, "rounds": 5}, False)
    assert load_or_init(p, {**CFG, "rounds": 5}, True)["results"] == []
    p.write_text("{not json")
    with pytest.raises(SystemExit):                      # corrupt -> never silently overwrite
        load_or_init(p, CFG, False)
    p.write_text(json.dumps([{"id": "old"}]))
    with pytest.raises(SystemExit):                      # legacy list format
        load_or_init(p, CFG, False)
