"""End-to-end FEVER experiment: a 2x2 design over (evidence) x (debate).

    single_llm           no evidence, no debate
    debate_only          no evidence, debate
    evidence_debate      evidence,    debate

All three conditions use the same model and the same final decision prompt
(JudgeService), so differences are attributable to evidence and/or debate.

Claims are drawn at random (seeded), stratified by gold label, from validation claims whose
gold evidence is inside the retrieval corpus. That restriction is a deliberate, documented
selection bias: it measures verification given *reachable* evidence, not full-FEVER accuracy.

Usage:  python run_verification_experiment.py --num-claims 50 --seed 0
"""
import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from config import DEFAULT_BASE_URL, DEFAULT_MODEL, LABELS
from services.corpus import corpus_ids, get_gold_ids
from services.debate_service import DebateService
from services.judge_service import JudgeService
from services.llm_client import LLMClient

PARSE_ERROR = "PARSE_ERROR"
CONDITIONS = ("single_llm", "debate_only", "evidence_debate")
CONFIG_KEYS = ("model", "rounds", "agents", "top_k", "seed", "num_claims", "agent_temperature",
               "judge_temperature", "include_title", "corpus_max_claims")

logger = logging.getLogger("experiment")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--num-claims", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--agent-temperature", type=float, default=0.7)
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--corpus-max-claims", type=int, default=50000)
    p.add_argument("--include-title", action="store_true")
    p.add_argument("--output", default="verification_experiment_50.json")
    p.add_argument("--sleep", type=float, default=2.0, help="seconds between claims")
    p.add_argument("--overwrite", action="store_true",
                   help="discard an existing results file with a different config")
    return p.parse_args()


# --------------------------------------------------------------------------- sampling
def select_claims(dataset, in_corpus, n: int, seed: int) -> List[Dict]:
    """Seeded, label-stratified sample of validation claims with gold evidence in the corpus."""
    by_label = defaultdict(list)
    for row in dataset:
        if get_gold_ids(row) & in_corpus and row["label"] in LABELS:
            by_label[row["label"]].append(row)

    rng = random.Random(seed)
    base, extra = divmod(n, len(LABELS))
    chosen = []
    for i, label in enumerate(LABELS):
        want = base + (1 if i < extra else 0)
        pool = by_label[label]
        if len(pool) < want:
            raise RuntimeError(f"Only {len(pool)} covered '{label}' claims, need {want}.")
        chosen.extend(rng.sample(pool, want))
    rng.shuffle(chosen)
    return chosen


# ------------------------------------------------------------------------ persistence
def save_atomic(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_or_init(path: Path, config: Dict, overwrite: bool) -> Dict:
    if not path.exists():
        return {"config": config, "results": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is corrupt ({exc}). Fix or delete it; refusing to overwrite.")
    if isinstance(data, list) or "config" not in data:
        raise SystemExit(f"{path} is in the legacy format. Use a different --output.")
    diff = {k: (data["config"].get(k), config[k]) for k in CONFIG_KEYS
            if data["config"].get(k) != config[k]}
    if diff and not overwrite:
        raise SystemExit(f"Config differs from the saved run: {diff}. "
                         "Use a new --output or pass --overwrite.")
    return {"config": config, "results": []} if diff else data


# ---------------------------------------------------------------------------- one claim
def verdict_of(result) -> str:
    return result.verdict if result.parse_ok else PARSE_ERROR


def condition_record(result, **extra) -> Dict:
    return {"verdict": verdict_of(result), "confidence": result.confidence,
            "explanation": result.explanation, "evidence_used": result.evidence_used,
            "grounded": result.grounded, "raw_response": result.raw, **extra}


def run_claim(row, retriever, debate: DebateService, judge: JudgeService, args) -> Dict:
    claim = row["claim"]
    gold = sorted(get_gold_ids(row))
    evidence = retriever.retrieve(claim, top_k=args.top_k)
    retrieved_ids = [(e["title"], str(e["sentence_id"])) for e in evidence]
    hit = bool(set(gold) & set(retrieved_ids))

    rec = {"id": str(row["id"]), "claim": claim, "gold_label": row["label"],
           "gold_evidence_ids": [list(g) for g in gold],
           "retrieved_ids": [list(r) for r in retrieved_ids],
           "gold_hit_at_k": hit, "evidence": evidence}

    rec["single_llm"] = condition_record(judge.judge(claim, [], None, use_evidence=False))

    hist = debate.run_debate(claim, evidence=None, rounds=args.rounds)
    rec["debate_only"] = condition_record(
        judge.judge(claim, [], hist, use_evidence=False), debate_history=hist)


    hist = debate.run_debate(claim, evidence=evidence, rounds=args.rounds)
    rec["evidence_debate"] = condition_record(
        judge.judge(claim, evidence, hist), debate_history=hist)

    rec["correct"] = {c: rec[c]["verdict"] == row["label"] for c in CONDITIONS}
    return rec


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set.")

    config = {k: getattr(args, k.replace("-", "_"), None) for k in CONFIG_KEYS}
    config["corpus_max_claims"] = args.corpus_max_claims
    out = Path(args.output)
    state = load_or_init(out, config, args.overwrite)
    done = {r["id"] for r in state["results"]}

    from datasets import load_dataset
    from services.corpus import load_fever_evidence
    from services.evidence_retriever import FeverEvidenceRetriever

    validation = load_dataset("copenlu/fever_gold_evidence", split="validation")
    evidence = load_fever_evidence(args.corpus_max_claims)
    retriever = FeverEvidenceRetriever(max_claims=args.corpus_max_claims,
                                       include_title=args.include_title, evidence=evidence)
    claims = select_claims(validation, corpus_ids(evidence), args.num_claims, args.seed)

    llm = LLMClient(api_key, args.model, args.base_url)
    debate = DebateService(llm, agents=args.agents, temperature=args.agent_temperature)
    judge = JudgeService(llm, temperature=args.judge_temperature)

    for i, row in enumerate(claims, 1):
        if str(row["id"]) in done:
            continue
        logger.info("claim %d/%d (gold=%s)", i, len(claims), row["label"])
        state["results"].append(run_claim(row, retriever, debate, judge, args))
        save_atomic(out, state)            # a failed claim is simply retried on resume
        time.sleep(args.sleep)

    logger.info("Done: %d claims saved to %s. Run: python compute_metrics.py %s",
                len(state["results"]), out, out)


if __name__ == "__main__":
    main()
