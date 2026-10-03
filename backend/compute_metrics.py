"""Metrics for verification_experiment results (new 2x2 format or the legacy list format).

Usage: python compute_metrics.py verification_experiment_v2.json [--json metrics.json]
"""
import argparse
import json
import math
from typing import Dict, List, Sequence

from config import LABELS
from services.retrieval_eval import wilson_interval

ALL_CONDITIONS = ("single_llm", "debate_only","evidence_debate")
PAIRS = (("single_llm", "debate_only"), ("single_llm", "evidence_debate"), ("debate_only", "evidence_debate"))


def prf(gold: Sequence[str], pred: Sequence[str], labels=LABELS) -> Dict:
    """Per-label and macro/weighted precision, recall, F1 (zero_division -> 0).
    Predictions outside `labels` (e.g. PARSE_ERROR) count as wrong."""
    per = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[lab] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
    n = len(gold)
    out = {"per_label": per}
    for name, w in (("macro", {l: 1 / len(labels) for l in labels}),
                    ("weighted", {l: per[l]["support"] / n if n else 0 for l in labels})):
        out[name] = {m: sum(w[l] * per[l][m] for l in labels)
                     for m in ("precision", "recall", "f1")}
    return out


def confusion(gold, pred, labels=LABELS):
    cols = list(labels) + ["OTHER"]
    m = {g: {c: 0 for c in cols} for g in labels}
    for g, p in zip(gold, pred):
        m[g][p if p in labels else "OTHER"] += 1
    return m


def mcnemar_exact(correct_a: List[bool], correct_b: List[bool]) -> Dict:
    b = sum(x and not y for x, y in zip(correct_a, correct_b))   # A right, B wrong
    c = sum(y and not x for x, y in zip(correct_a, correct_b))   # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def load_results(path: str):
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        return None, data
    return data.get("config"), data["results"]


def compute(results: List[Dict]) -> Dict:
    conds = [c for c in ALL_CONDITIONS if results and c in results[0]]
    gold = [r["gold_label"] for r in results]
    out = {"n": len(results), "conditions": {}, "pairwise_mcnemar": {}}
    for c in conds:
        pred = [r[c]["verdict"] for r in results]
        k = sum(g == p for g, p in zip(gold, pred))
        entry = {"accuracy": k / len(gold), "accuracy_ci95": list(wilson_interval(k, len(gold))),
                 "parse_errors": sum(p not in LABELS for p in pred),
                 "ungrounded": sum(not r[c].get("grounded", True) for r in results),
                 "prediction_counts": {l: pred.count(l) for l in set(pred)},
                 **prf(gold, pred), "confusion": confusion(gold, pred)}
        out["conditions"][c] = entry
    for a, b in PAIRS:
        if a in conds and b in conds:
            out["pairwise_mcnemar"][f"{a} vs {b}"] = mcnemar_exact(
                [r[a]["verdict"] == r["gold_label"] for r in results],
                [r[b]["verdict"] == r["gold_label"] for r in results])
    if results and "gold_hit_at_k" in results[0]:
        hit = [r["gold_hit_at_k"] for r in results]
        out["retrieval_hit_at_k"] = sum(hit) / len(hit)
        for c in ( "evidence_debate", ):
            if c in conds:
                for flag, name in ((True, "when_gold_retrieved"), (False, "when_gold_missed")):
                    sub = [r[c]["verdict"] == r["gold_label"] for r in results
                           if r["gold_hit_at_k"] == flag]
                    out["conditions"][c][f"accuracy_{name}"] = (sum(sub) / len(sub)) if sub else None
                    out["conditions"][c][f"n_{name}"] = len(sub)
    return out


def print_report(m: Dict) -> None:
    print(f"\nClaims: {m['n']}\n")
    print(f"{'condition':<22}{'acc':>7}{'95% CI':>16}{'macroP':>8}{'macroR':>8}{'macroF1':>9}"
          f"{'wF1':>7}{'parse_err':>10}")
    for c, e in m["conditions"].items():
        lo, hi = e["accuracy_ci95"]
        print(f"{c:<22}{e['accuracy']*100:>6.1f}%{f'[{lo*100:.0f}-{hi*100:.0f}]':>16}"
              f"{e['macro']['precision']*100:>7.1f}%{e['macro']['recall']*100:>7.1f}%"
              f"{e['macro']['f1']*100:>8.1f}%{e['weighted']['f1']*100:>6.1f}%{e['parse_errors']:>10}")
    if "retrieval_hit_at_k" in m:
        print(f"\nGold evidence in retrieved top-k: {m['retrieval_hit_at_k']*100:.1f}%")
        for c in ("evidence_debate", ):
            e = m["conditions"].get(c)
            if e:
                print(f"  {c}: acc when gold retrieved = {e['accuracy_when_gold_retrieved']} "
                      f"(n={e['n_when_gold_retrieved']}), when missed = "
                      f"{e['accuracy_when_gold_missed']} (n={e['n_when_gold_missed']})")
    print("\nMcNemar exact (paired accuracy differences):")
    for k, v in m["pairwise_mcnemar"].items():
        print(f"  {k}: {v['a_only_correct']} vs {v['b_only_correct']} discordant, p={v['p_value']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--json", help="also write metrics to this file")
    a = ap.parse_args()
    _, res = load_results(a.results)
    metrics = compute(res)
    print_report(metrics)
    if a.json:
        json.dump(metrics, open(a.json, "w", encoding="utf-8"), indent=2)
