import pytest

from services.debate_service import DebateService


class ScriptedLLM:
    """Replies '<<n>>' to the n-th call and records the prompt it was sent."""
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=0.0):
        n = len(self.calls) + 1
        self.calls.append({"messages": [dict(m) for m in messages], "temperature": temperature})
        return f"<<{n}>>"


def visible_replies(call):
    last = call["messages"][-1]["content"]
    return {i for i in range(1, 20) if f"<<{i}>>" in last}


def test_round_2_and_3_see_only_previous_round_of_other_agents():
    """Regression: agent 2 used to see agent 1's SAME-round answer."""
    llm = ScriptedLLM()
    DebateService(llm, agents=2).run_debate("c", rounds=3)
    seen = [visible_replies(c) for c in llm.calls]
    # call order: R1A1=1 R1A2=2 | R2A1=3 R2A2=4 | R3A1=5 R3A2=6
    assert seen[0] == set() and seen[1] == set()          # round 1 fully independent
    assert seen[2] == {2} and seen[3] == {1}              # round 2: previous-round answers
    assert seen[4] == {4} and seen[5] == {3}              # round 3: round-2 answers only


def test_agent_never_sees_own_answer_in_the_other_agents_block_and_contexts_are_separate():
    llm = ScriptedLLM()
    DebateService(llm, agents=2).run_debate("c", rounds=2)
    a1_r2 = llm.calls[2]["messages"]
    assert [m["role"] for m in a1_r2] == ["user", "assistant", "user"]
    assert a1_r2[1]["content"] == "<<1>>"                  # own history preserved
    assert "<<1>>" not in a1_r2[2]["content"]              # own answer not echoed as "other"


def test_three_agents_each_see_both_others():
    llm = ScriptedLLM()
    DebateService(llm, agents=3).run_debate("c", rounds=2)
    assert visible_replies(llm.calls[3]) == {2, 3}
    assert visible_replies(llm.calls[4]) == {1, 3}
    assert visible_replies(llm.calls[5]) == {1, 2}


def test_history_shape_and_counts():
    llm = ScriptedLLM()
    hist = DebateService(llm, agents=2, temperature=0.7).run_debate("c", rounds=3)
    assert [h["round"] for h in hist] == [1, 2, 3]
    assert all(set(h) == {"round", "agent_1", "agent_2"} for h in hist)
    assert len(llm.calls) == 6 and all(c["temperature"] == 0.7 for c in llm.calls)


def test_evidence_only_appears_when_provided():
    llm = ScriptedLLM()
    ev = [{"title": "T", "sentence_id": "0", "text": "hello evidence"}]
    DebateService(llm).run_debate("c", evidence=ev, rounds=1)
    DebateService(llm).run_debate("c", evidence=None, rounds=1)
    assert "hello evidence" in llm.calls[0]["messages"][0]["content"]
    assert "Retrieved evidence" not in llm.calls[2]["messages"][0]["content"]


def test_invalid_config():
    with pytest.raises(ValueError):
        DebateService(ScriptedLLM(), agents=1)
    with pytest.raises(ValueError):
        DebateService(ScriptedLLM()).run_debate("c", rounds=0)
