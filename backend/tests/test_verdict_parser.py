
from services.verdict_service import (parse_confidence, parse_evidence_used,
                                      parse_judge_response, parse_verdict)

GOOD = """VERDICT: REFUTES

CONFIDENCE: 0.85

EXPLANATION: Evidence 2 says the opposite.
It is clear.

EVIDENCE_USED: Evidence 2, Evidence 9"""


def test_full_parse_and_evidence_validation():
    r = parse_judge_response(GOOD, n_evidence=5)
    assert r.parse_ok and r.verdict == "REFUTES" and r.confidence == 0.85
    assert r.explanation.startswith("Evidence 2 says") and "clear." in r.explanation
    assert r.evidence_used == [2] and r.invalid_evidence_refs == [9]
    assert r.grounded


def test_markdown_bold_and_case():
    assert parse_verdict("**VERDICT:** **Not Enough Info**") == "NOT ENOUGH INFO"
    assert parse_verdict("verdict: not_enough_info") == "NOT ENOUGH INFO"


def test_missing_or_ambiguous_verdict_is_a_parse_failure_not_nei():
    assert parse_verdict("I think it is true.") is None
    assert parse_verdict("VERDICT: SUPPORTS / REFUTES / NOT ENOUGH INFO") is None
    r = parse_judge_response("no structure here", 5)
    assert not r.parse_ok and r.verdict is None and not r.grounded


def test_confidence_variants():
    assert parse_confidence("CONFIDENCE: 1") == 1.0
    assert parse_confidence("CONFIDENCE: 85%") == 0.85
    assert parse_confidence("CONFIDENCE: 7") is None
    assert parse_confidence("nothing") is None


def test_supports_without_citation_is_ungrounded():
    r = parse_judge_response("VERDICT: SUPPORTS\nCONFIDENCE: 0.9\nEXPLANATION: x\nEVIDENCE_USED: None", 5)
    assert r.parse_ok and not r.grounded
    r2 = parse_judge_response("VERDICT: NOT ENOUGH INFO\nEVIDENCE_USED: None", 5)
    assert r2.grounded


def test_evidence_used_bounds():
    assert parse_evidence_used("EVIDENCE_USED: 0, 1, 6", 5) == ([1], [0, 6])
