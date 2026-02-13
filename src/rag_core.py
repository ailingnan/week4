import re, time, csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

# ✅ FAISS is optional (Streamlit Cloud often can't install it)
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception as e:
    HAS_FAISS = False

@dataclass
class SubChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str

@dataclass
class RetrievalResult:
    score: float
    chunk: SubChunk

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for i in range(len(doc)):
            t = doc.load_page(i).get_text("text") or ""
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                pages.append((i+1, t))
    return pages

def fixed_size_chunk(text: str, words_per_chunk: int = 250, overlap: int = 40) -> List[str]:
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return out

class Week3RAG:
    def __init__(self, docs_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_dir = Path(docs_dir)
        self.embedder = SentenceTransformer(model_name)
        self.sub_chunks: List[SubChunk] = []
        self.vecs: np.ndarray | None = None
        self.index = None  # faiss index if available

    def build(self):
        pdfs = sorted(self.docs_dir.rglob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found under: {self.docs_dir}")

        subs: List[SubChunk] = []
        for pdf_path in pdfs:
            doc_id = pdf_path.name
            for page_num, page_text in extract_pdf_pages(str(pdf_path)):
                for j, t in enumerate(fixed_size_chunk(page_text, words_per_chunk=250, overlap=40)):
                    subs.append(SubChunk(
                        chunk_id=f"{doc_id}::p{page_num}::c{j+1}",
                        doc_id=doc_id,
                        page_num=page_num,
                        text=t
                    ))
        self.sub_chunks = subs

        texts = [c.text for c in self.sub_chunks]
        vecs = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        self.vecs = vecs

        # ✅ Build FAISS index only if available
        if HAS_FAISS:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            self.index = index
        else:
            self.index = None

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if self.vecs is None or len(self.sub_chunks) == 0:
            return []

        qv = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

        if HAS_FAISS and self.index is not None:
            scores, idxs = self.index.search(qv, top_k)
            scores = scores[0]
            idxs = idxs[0]
        else:
            # ✅ numpy cosine (since embeddings are normalized, dot = cosine)
            sims = (self.vecs @ qv[0]).astype(np.float32)
            idxs = np.argsort(-sims)[:top_k]
            scores = sims[idxs]

        # critical fix: if all scores ~0 -> no evidence
        if float(np.max(scores)) <= 1e-8:
            return []

        out = []
        for s, i in zip(scores, idxs):
            if float(s) > 0:
                out.append(RetrievalResult(score=float(s), chunk=self.sub_chunks[int(i)]))
        return out

    def build_evidence(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        return [{
            "evidence_id": r.chunk.chunk_id,
            "source": r.chunk.doc_id,
            "page": r.chunk.page_num,
            "score": r.score,
            "text": r.chunk.text
        } for r in results]

    def generate_answer_fallback(self, query: str, evidence: List[Dict[str, Any]]) -> Tuple[str, float]:
        if not evidence:
            return "I don't have enough information in the provided documents to answer that.", 0.0
        top = evidence[0]
        answer = f"Based on {top['source']} (p.{top['page']}) [Evidence: {top['evidence_id']}]: {top['text'][:350]}..."
        confidence = float(max(0.1, min(0.95, top["score"])))
        return answer, confidence

    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        t0 = time.time()
        results = self.retrieve(query, top_k=top_k)
        evidence = self.build_evidence(results)
        answer, confidence = self.generate_answer_fallback(query, evidence)
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "answer": answer,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "evidence": evidence,
            "evidence_ids": [e["evidence_id"] for e in evidence],
        }

# Logging (required)
LOG_PATH = Path("logs/product_metrics.csv")
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

def log_interaction(query: str, latency_ms: float, evidence_ids: List[str], confidence: float):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp","query","latency_ms","evidence_ids","confidence"])
        from datetime import datetime
        w.writerow([datetime.now().isoformat(timespec="seconds"), query, f"{latency_ms:.0f}", ";".join(evidence_ids), f"{confidence:.2f}"])

def rag_query_logged(rag: Week3RAG, query: str, top_k: int = 5) -> Dict[str, Any]:
    resp = rag.query(query, top_k=top_k)
    log_interaction(query, resp["latency_ms"], resp["evidence_ids"], resp["confidence"])
    return resp
