"""Parsing and validation of the judge's structured output.

Parse failures are reported explicitly (parse_ok=False, verdict=None) instead of being
silently mapped to NOT ENOUGH INFO, so they can never masquerade as a real verdict.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from config import LABELS

_LABEL_RE = re.compile(r"NOT[ _]ENOUGH[ _]INFO|SUPPORTS|REFUTES")


@dataclass
class JudgeResult:
    verdict: Optional[str]
    confidence: Optional[float]
    explanation: str
    evidence_used: List[int] = field(default_factory=list)      # valid 1-based indices
    invalid_evidence_refs: List[int] = field(default_factory=list)
    grounded: bool = True   # SUPPORTS/REFUTES must cite at least one valid evidence item
    parse_ok: bool = True
    raw: str = ""


def _clean(text: str) -> str:
    return text.replace("*", "").replace("`", "")


def _field_line(text: str, name: str) -> Optional[str]:
    m = re.search(rf"^\s*{name}\s*:(.*)$", _clean(text), re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_verdict(text: str) -> Optional[str]:
    line = _field_line(text, "VERDICT")
    if line is None:
        return None
    labels = {re.sub(r"_", " ", m) for m in _LABEL_RE.findall(line.upper())}
    # Echoing the template ("SUPPORTS / REFUTES / ...") is not a decision.
    return labels.pop() if len(labels) == 1 else None


def parse_confidence(text: str) -> Optional[float]:
    line = _field_line(text, "CONFIDENCE")
    if line is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%?)", line)
    if not m:
        return None
    value = float(m.group(1)) / (100.0 if m.group(2) else 1.0)
    return value if 0.0 <= value <= 1.0 else None


def parse_explanation(text: str) -> str:
    m = re.search(r"EXPLANATION\s*:\s*(.*?)(?:\n\s*EVIDENCE_USED\s*:|\Z)",
                  _clean(text), re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_evidence_used(text: str, n_evidence: int):
    line = _field_line(text, "EVIDENCE_USED") or ""
    numbers = [int(n) for n in re.findall(r"\d+", line)]
    valid = sorted({n for n in numbers if 1 <= n <= n_evidence})
    invalid = sorted({n for n in numbers if not 1 <= n <= n_evidence})
    return valid, invalid


def parse_judge_response(text: str, n_evidence: int) -> JudgeResult:
    verdict = parse_verdict(text)
    valid, invalid = parse_evidence_used(text, n_evidence)
    grounded = verdict == "NOT ENOUGH INFO" or bool(valid)
    return JudgeResult(
        verdict=verdict if verdict in LABELS else None,
        confidence=parse_confidence(text),
        explanation=parse_explanation(text),
        evidence_used=valid,
        invalid_evidence_refs=invalid,
        grounded=grounded if verdict else False,
        parse_ok=verdict is not None,
        raw=text,
    )
