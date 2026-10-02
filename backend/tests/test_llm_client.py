import openai
import pytest

from services.llm_client import LLMClient, LLMError


def make_client(sleeps):
    return LLMClient("k", "m", "http://x", max_retries=3, sleep=sleeps.append)


def ok(text="hi"):
    return {"choices": [{"message": {"content": text}}]}


def test_retries_transient_errors_with_backoff(monkeypatch):
    seq = [openai.error.Timeout("t"), openai.error.RateLimitError("429"), ok("done")]

    def fake(**kw):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(openai.ChatCompletion, "create", staticmethod(fake))
    sleeps = []
    assert make_client(sleeps).ask("p") == "done"
    assert sleeps == [5.0, 10.0]


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(openai.ChatCompletion, "create",
                        staticmethod(lambda **kw: (_ for _ in ()).throw(openai.error.Timeout("t"))))
    with pytest.raises(LLMError):
        make_client([]).ask("p")


def test_non_retryable_fails_fast_and_empty_reply_is_an_error(monkeypatch):
    calls = []

    def bad(**kw):
        calls.append(1)
        raise openai.error.InvalidRequestError("bad", "x")
    monkeypatch.setattr(openai.ChatCompletion, "create", staticmethod(bad))
    with pytest.raises(LLMError):
        make_client([]).ask("p")
    assert len(calls) == 1
    monkeypatch.setattr(openai.ChatCompletion, "create", staticmethod(lambda **kw: ok("  ")))
    with pytest.raises(LLMError):
        make_client([]).ask("p")


def test_credentials_are_per_request_not_global(monkeypatch):
    seen = {}
    monkeypatch.setattr(openai.ChatCompletion, "create",
                        staticmethod(lambda **kw: seen.update(kw) or ok()))
    make_client([]).ask("p")
    assert seen["api_key"] == "k" and seen["api_base"] == "http://x" and seen["request_timeout"] == 90
