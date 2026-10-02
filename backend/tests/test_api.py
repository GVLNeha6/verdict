import pytest
from fastapi.testclient import TestClient

from main import create_app
from services.llm_client import LLMError
from services.pipeline import JudgeParseError

RESP = {"claim": "x", "verdict": "SUPPORTS", "confidence": 0.9, "explanation": "e",
        "evidence": [{"title": "T", "sentence_id": "1", "text": "t", "score": 0.5}],
        "evidence_used": [1], "grounded": True,
        "debate": {"agents": 2, "rounds": 3, "final_reasoning": {"agent_1": "a", "agent_2": "b"}}}


class FakePipeline:
    def __init__(self, exc=None):
        self.exc = exc

    def verify(self, claim):
        if self.exc:
            raise self.exc
        return {**RESP, "claim": claim}


def client(exc=None, api_key=None):
    return TestClient(create_app(lambda: FakePipeline(exc), api_key=api_key))


def test_success_and_schema():
    with client() as c:
        r = c.post("/api/verify", json={"claim": "  hello  "})
        assert r.status_code == 200 and r.json()["claim"] == "hello"
        assert c.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("body", [{"claim": ""}, {"claim": "   "}, {}, {"claim": "x" * 501}])
def test_validation_rejects_bad_input(body):
    with client() as c:
        assert c.post("/api/verify", json=body).status_code == 422


@pytest.mark.parametrize("exc,code", [(LLMError("boom"), 502), (JudgeParseError("bad"), 502),
                                      (RuntimeError("secret internals"), 500)])
def test_errors_are_mapped_and_do_not_leak(exc, code):
    with client(exc) as c:
        r = c.post("/api/verify", json={"claim": "x"})
        assert r.status_code == code and "secret" not in r.text and "boom" not in r.text


def test_optional_api_key_auth():
    with client(api_key="s3cret") as c:
        assert c.post("/api/verify", json={"claim": "x"}).status_code == 401
        assert c.post("/api/verify", json={"claim": "x"}, headers={"X-API-Key": "no"}).status_code == 401
        assert c.post("/api/verify", json={"claim": "x"}, headers={"X-API-Key": "s3cret"}).status_code == 200
