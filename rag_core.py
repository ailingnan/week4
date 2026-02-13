
import re, time, csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import fitz  # pymupdf
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------
# Data structures
# -------------------------
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

# -------------------------
# PDF -> text
# -------------------------
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

# -------------------------
# Core RAG Engine
# -------------------------
class Week3RAG:
    def __init__(self, docs_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_dir = Path(docs_dir)
        self.embedder = SentenceTransformer(model_name)
        self.sub_chunks: List[SubChunk] = []
        self.vecs: np.ndarray = None
        self.index = None

    def build(self):
        pdfs = sorted(self.docs_dir.rglob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found under: {self.docs_dir}")

        # 1) Extract + chunk
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

        # 2) Embed
        texts = [c.text for c in self.sub_chunks]
        vecs = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        self.vecs = vecs

        # 3) FAISS (IP for normalized embeddings)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        qv = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, idxs = self.index.search(qv, top_k)

        # critical fix: if all scores ~ 0 -> return empty evidence
        if float(np.max(scores)) <= 1e-8:
            return []

        out = []
        for s, i in zip(scores[0], idxs[0]):
            if int(i) >= 0 and float(s) > 0:
                out.append(RetrievalResult(score=float(s), chunk=self.sub_chunks[int(i)]))
        return out

    def build_evidence(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        ev = []
        for r in results:
            ev.append({
                "evidence_id": r.chunk.chunk_id,
                "source": r.chunk.doc_id,
                "page": r.chunk.page_num,
                "score": r.score,
                "text": r.chunk.text
            })
        return ev

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

# -------------------------
# Logging (Week4 requirement)
# -------------------------
LOG_PATH = Path("logs/product_metrics.csv")
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

def log_interaction(query: str, latency_ms: float, evidence_ids: List[str], confidence: float):
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp","query","latency_ms","evidence_ids","confidence"])
        from datetime import datetime
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            f"{latency_ms:.0f}",
            ";".join(evidence_ids),
            f"{confidence:.2f}"
        ])

def rag_query_logged(rag: Week3RAG, query: str, top_k: int = 5) -> Dict[str, Any]:
    resp = rag.query(query, top_k=top_k)
    log_interaction(query, resp["latency_ms"], resp["evidence_ids"], resp["confidence"])
    return resp
