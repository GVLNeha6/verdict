"""FEVER evidence corpus construction and shared text utilities.

Every retriever (SBERT+FAISS, TF-IDF, BM25) builds on exactly the same corpus and the same
tokenisation, so retrieval comparisons differ only in the ranking algorithm.
"""
import hashlib
import re
from typing import Dict, Iterable, List, Set, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"\w+", re.UNICODE)

_TITLE_FIXES = (("-LRB-", "("), ("-RRB-", ")"), ("-COLON-", ":"), ("_", " "))


def tokenize(text: str) -> List[str]:
    """Lower-case word tokens with English stop-words removed (used by TF-IDF and BM25)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def clean_title(title: str) -> str:
    for old, new in _TITLE_FIXES:
        title = title.replace(old, new)
    return title


def index_text(item: Dict, include_title: bool) -> str:
    """Text that is embedded / tokenised for one evidence record."""
    if include_title:
        return f"{clean_title(item['title'])}: {item['text']}"
    return item["text"]


def load_fever_evidence(max_claims: int = 50000) -> List[Dict]:
    """Unique gold-evidence sentences of the first `max_claims` FEVER training claims."""
    from datasets import load_dataset

    train = load_dataset("copenlu/fever_gold_evidence")["train"]
    data = train.select(range(min(max_claims, len(train))))

    unique: Dict[Tuple[str, str, str], Dict] = {}
    for row in data:
        for item in row["evidence"]:
            if len(item) < 3:
                continue
            title, sentence_id, text = item[0], str(item[1]), item[2]
            unique.setdefault((title, sentence_id, text),
                              {"title": title, "sentence_id": sentence_id, "text": text})
    return list(unique.values())


def get_gold_ids(row) -> Set[Tuple[str, str]]:
    """(title, sentence_id) pairs of a FEVER row's gold evidence, ids normalised to str."""
    return {(item[0], str(item[1])) for item in row["evidence"] if len(item) >= 2}


def corpus_ids(evidence: Iterable[Dict]) -> Set[Tuple[str, str]]:
    return {(item["title"], str(item["sentence_id"])) for item in evidence}


def corpus_fingerprint(evidence: List[Dict]) -> str:
    h = hashlib.sha1()
    for item in evidence:
        h.update(f"{item['title']}\t{item['sentence_id']}\t{item['text']}\n".encode("utf-8"))
    return h.hexdigest()
