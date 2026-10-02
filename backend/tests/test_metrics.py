import pytest

from compute_metrics import compute, confusion, mcnemar_exact, prf
from services.retrieval_eval import wilson_interval

S, R, N = "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
# The 10-claim run reported earlier (gold, single, debate_only, evidence_debate).
OLD = [(R, S, N, R), (N, R, N, N), (R, N, N, R), (R, R, N, R), (N, N, N, N),
       (N, S, N, N), (R, R, N, R), (S, S, N, S), (S, R, N, N), (N, S, N, N)]


def rows():
    return [{"gold_label": g, "single_llm": {"verdict": a}, "debate_only": {"verdict": b},
             "evidence_debate": {"verdict": c}} for g, a, b, c in OLD]


def test_reproduces_previously_reported_numbers():
    m = compute(rows())["conditions"]
    assert m["single_llm"]["accuracy"] == 0.4 and round(m["single_llm"]["macro"]["f1"], 4) == 0.3889
    assert m["debate_only"]["accuracy"] == 0.4 and round(m["debate_only"]["macro"]["f1"], 4) == 0.1905
    assert round(m["debate_only"]["weighted"]["f1"], 4) == 0.2286
    assert m["evidence_debate"]["accuracy"] == 0.9 and round(m["evidence_debate"]["macro"]["f1"], 4) == 0.8519
    assert round(m["evidence_debate"]["weighted"]["precision"], 4) == 0.92


def test_parse_errors_count_as_wrong_and_are_reported():
    e = compute([{"gold_label": S, "single_llm": {"verdict": "PARSE_ERROR"}}])["conditions"]["single_llm"]
    assert e["accuracy"] == 0 and e["parse_errors"] == 1
    assert confusion([S], ["PARSE_ERROR"])[S]["OTHER"] == 1


def test_mcnemar_exact():
    assert mcnemar_exact([True] * 5, [True] * 5)["p_value"] == 1.0
    r = mcnemar_exact([True] * 8 + [False] * 2, [False] * 8 + [True] * 2)   # b=8, c=2
    assert r["p_value"] == pytest.approx(0.109375)


def test_wilson_matches_known_values():
    lo, hi = wilson_interval(9, 10)
    assert (round(lo, 2), round(hi, 2)) == (0.60, 0.98)
    assert wilson_interval(0, 0) == (0.0, 0.0)
