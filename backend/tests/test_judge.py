from services.judge_service import JudgeService

OK = "VERDICT: SUPPORTS\nCONFIDENCE: 0.9\nEXPLANATION: e\nEVIDENCE_USED: Evidence 1"
EV = [{"title": "T", "sentence_id": "0", "text": "fact"}]
HIST = [{"round": 1, "agent_1": "A1 text", "agent_2": "A2 text", "agent_3": "A3 text"}]


class LLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def ask(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_no_evidence_means_no_llm_call_and_nei():
    llm = LLM([])
    r = JudgeService(llm).judge("c", [], HIST, use_evidence=True)
    assert r.verdict == "NOT ENOUGH INFO" and llm.prompts == []


def test_all_agents_reach_the_judge():
    llm = LLM([OK])
    JudgeService(llm).judge("c", EV, HIST)
    assert all(t in llm.prompts[0] for t in ("A1 text", "A2 text", "A3 text"))


def test_evidence_and_no_evidence_prompts_differ_as_intended():
    llm = LLM([OK, OK.replace("SUPPORTS", "REFUTES")])
    j = JudgeService(llm)
    j.judge("c", EV, None, use_evidence=True)
    j.judge("c", [], None, use_evidence=False)
    assert "Do not use outside knowledge" in llm.prompts[0] and "fact" in llm.prompts[0]
    assert "own knowledge" in llm.prompts[1] and "Do not use outside knowledge" not in llm.prompts[1]
    assert "FINAL AGENT" not in llm.prompts[0]          # baseline has no debate section


def test_retry_then_success_and_final_failure():
    llm = LLM(["garbage", OK])
    assert JudgeService(llm, parse_retries=1).judge("c", EV).parse_ok
    llm = LLM(["garbage", "still garbage"])
    r = JudgeService(llm, parse_retries=1).judge("c", EV)
    assert not r.parse_ok and r.verdict is None
