"""Full pipeline / experiment loop with a fake LLM and fake retriever."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from compute_metrics import compute
from config import Settings
from main import create_app
from run_verification_experiment import CONDITIONS, run_claim
from services.debate_service import DebateService
from services.judge_service import JudgeService
from services.pipeline import VerificationPipeline

EV = [{"title": "Paris", "sentence_id": 1, "text": "Paris is in France.", "score": 0.8},
      {"title": "Lyon", "sentence_id": "2", "text": "Lyon is in France.", "score": 0.6}]


class FakeLLM:
    def __init__(self, judge_reply):
        self.judge_reply, self.n_chat, self.n_ask = judge_reply, 0, 0

    def chat(self, messages, temperature=0.0):
        self.n_chat += 1
        return "reasoning\nCURRENT VERDICT: SUPPORTS"

    def ask(self, prompt, temperature=0.0):
        self.n_ask += 1
        return self.judge_reply


class FakeRetriever:
    def retrieve(self, claim, top_k=5, min_score=0.0):
        return [dict(e) for e in EV[:top_k]]


JUDGE = "VERDICT: SUPPORTS\nCONFIDENCE: 0.9\nEXPLANATION: ok\nEVIDENCE_USED: Evidence 1"


def make_pipeline(reply=JUDGE):
    llm = FakeLLM(reply)
    s = Settings(llm_api_key="k")
    return VerificationPipeline(s, FakeRetriever(), DebateService(llm, s.agents), JudgeService(llm)), llm


def test_api_end_to_end_response_shape():
    pipe, llm = make_pipeline()
    with TestClient(create_app(lambda: pipe)) as c:
        body = c.post("/api/verify", json={"claim": "Paris is in France"}).json()
    assert body["verdict"] == "SUPPORTS" and body["evidence_used"] == [1] and body["grounded"]
    assert body["evidence"][0]["sentence_id"] == "1"                 # int coerced to str
    assert body["debate"]["agents"] == 2 and body["debate"]["rounds"] == 3
    assert set(body["debate"]["final_reasoning"]) == {"agent_1", "agent_2"}
    assert llm.n_chat == 6 and llm.n_ask == 1


def test_unparseable_judge_becomes_502_not_a_fake_verdict():
    pipe, _ = make_pipeline("I refuse to follow the format")
    with TestClient(create_app(lambda: pipe)) as c:
        assert c.post("/api/verify", json={"claim": "x"}).status_code == 502


def test_experiment_row_has_three_conditions_and_metrics_work():
    pipe, llm = make_pipeline()
    args = SimpleNamespace(top_k=2, rounds=2)
    row = {"id": 7, "claim": "Paris is in France", "label": "SUPPORTS",
           "evidence": [["Paris", 1, "Paris is in France."]]}
    rec = run_claim(row, FakeRetriever(), pipe.debate, pipe.judge, args)

    assert all(c in rec for c in CONDITIONS)
    assert rec["gold_hit_at_k"] is True and rec["gold_evidence_ids"] == [["Paris", "1"]]
    assert rec["correct"]["evidence_debate"] is True
    assert "debate_history" in rec["debate_only"] and "debate_history" not in rec["single_llm"]
    assert llm.n_ask == 3 and llm.n_chat == 8                        # 4 judge calls; 2 debates x 2 agents x 2 rounds
    json.dumps(rec)                                                   # must be serialisable

    m = compute([rec])
    assert m["conditions"]["evidence_debate"]["accuracy"] == 1.0
    assert m["retrieval_hit_at_k"] == 1.0 and "single_llm vs debate_only" in m["pairwise_mcnemar"]
