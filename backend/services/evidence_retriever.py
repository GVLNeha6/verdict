"""Dense retriever: Sentence-BERT embeddings + exact cosine search with FAISS."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from services.corpus import corpus_fingerprint, index_text, load_fever_evidence

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "retriever_index"


class FeverEvidenceRetriever:
    def __init__(self, max_claims: int = 50000, include_title: bool = False,
                 evidence: Optional[List[Dict]] = None, index_dir: Optional[Path] = None,
                 model_name: str = MODEL_NAME):
        self.max_claims = max_claims
        self.include_title = include_title
        self.model_name = model_name
        self.index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "fever.index"
        self.evidence_path = self.index_dir / "evidence.json"
        self.meta_path = self.index_dir / "meta.json"

        logger.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)

        wanted = {"model": model_name, "max_claims": max_claims, "include_title": include_title}
        meta = self._read_meta()
        cache_matches = meta is not None and all(meta.get(k) == v for k, v in wanted.items())

        if evidence is None and cache_matches and self.index_path.exists() \
                and self.evidence_path.exists():
            self.evidence = json.loads(self.evidence_path.read_text(encoding="utf-8"))
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Loaded cached index with %d records", len(self.evidence))
            return

        self.evidence = evidence if evidence is not None else load_fever_evidence(max_claims)
        fingerprint = corpus_fingerprint(self.evidence)
        if cache_matches and meta.get("corpus_sha1") == fingerprint and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Reused cached index for identical corpus")
        else:
            self._build_index()
            self._save(wanted, fingerprint)

    # ----------------------------------------------------------------- caching
    def _read_meta(self) -> Optional[Dict]:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _save(self, wanted: Dict, fingerprint: str) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.evidence_path.write_text(json.dumps(self.evidence, ensure_ascii=False),
                                      encoding="utf-8")
        self.meta_path.write_text(json.dumps(
            {**wanted, "n_records": len(self.evidence), "corpus_sha1": fingerprint}),
            encoding="utf-8")

    def _build_index(self) -> None:
        texts = [index_text(e, self.include_title) for e in self.evidence]
        logger.info("Embedding %d evidence records", len(texts))
        embeddings = self.model.encode(texts, convert_to_numpy=True,
                                       show_progress_bar=True).astype("float32")
        faiss.normalize_L2(embeddings)          # inner product == cosine similarity
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    # --------------------------------------------------------------- retrieval
    def retrieve(self, claim: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        query = self.model.encode([claim], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < min_score:
                continue
            item = dict(self.evidence[idx])
            item["score"] = float(score)
            results.append(item)
        return results
