"""Central configuration. Everything tunable lives here or in environment variables."""
import os
from dataclasses import dataclass
from typing import Optional

MAX_CLAIM_CHARS = 500
LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    llm_timeout: int = 90
    agents: int = 2
    rounds: int = 3
    agent_temperature: float = 0.7   # >0 so agents are genuinely independent samples
    judge_temperature: float = 0.0
    top_k: int = 5
    min_evidence_score: float = 0.0  # 0 disables the filter; calibrate on validation data
    corpus_max_claims: int = 50000   # number of FEVER *claims* whose gold evidence is indexed
    include_title: bool = False
    api_key: Optional[str] = None    # optional shared secret for X-API-Key auth


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    env = os.getenv
    return Settings(
        llm_api_key=key,
        model=env("LLM_MODEL", DEFAULT_MODEL),
        base_url=env("LLM_BASE_URL", DEFAULT_BASE_URL),
        llm_timeout=int(env("LLM_TIMEOUT", "90")),
        agents=int(env("DEBATE_AGENTS", "2")),
        rounds=int(env("DEBATE_ROUNDS", "3")),
        agent_temperature=float(env("AGENT_TEMPERATURE", "0.7")),
        judge_temperature=float(env("JUDGE_TEMPERATURE", "0.0")),
        top_k=int(env("TOP_K", "5")),
        min_evidence_score=float(env("MIN_EVIDENCE_SCORE", "0.0")),
        corpus_max_claims=int(env("CORPUS_MAX_CLAIMS", "50000")),
        include_title=_bool(env("INCLUDE_TITLE", "false")),
        api_key=env("VERDICT_API_KEY") or None,
    )
