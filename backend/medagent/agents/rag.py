"""RAG Agent: document chunking, dual vector store (numpy/pinecone), and medical retrieval."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from medagent.config import settings
from medagent.schemas import CaseInput, ImagingReport, LabReport, RagReport, Evidence

logger = logging.getLogger("medagent.rag")

# In-memory index cache
_EMBEDDING_MODEL = None
_VECTORIZER_FALLBACK = None
_INDEX_CHUNKS: Optional[List[Dict[str, Any]]] = None
_INDEX_VECTORS: Optional[np.ndarray] = None
_BACKEND_NAME: str = "numpy"

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def get_embedding_model() -> Tuple[Any, str]:
    """Retrieve sentence-transformers model or fallback to TF-IDF."""
    global _EMBEDDING_MODEL, _VECTORIZER_FALLBACK, _BACKEND_NAME
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL, "sentence-transformers"
    if _VECTORIZER_FALLBACK is not None:
        return _VECTORIZER_FALLBACK, "tfidf"

    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)
        _BACKEND_NAME = "numpy"
        return _EMBEDDING_MODEL, "sentence-transformers"
    except Exception as e:
        logger.warning("sentence-transformers unavailable (%s); falling back to TF-IDF.", str(e))
        from sklearn.feature_extraction.text import TfidfVectorizer
        _VECTORIZER_FALLBACK = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        _BACKEND_NAME = "tfidf"
        return _VECTORIZER_FALLBACK, "tfidf"


def chunk_document(doc_path: Path) -> List[Dict[str, Any]]:
    """
    Split markdown/text documents on headings, then ~900 characters with 120 overlap,
    ending on sentence boundaries, assigning unique chunk IDs.
    """
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    doc_name = doc_path.stem

    # Split on headings (#, ##, ###)
    heading_pattern = re.compile(r"^(#{1,3}\s+[^\n]+)", re.MULTILINE)
    splits = heading_pattern.split(text)

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    current_section = "General"
    for part in splits:
        part_clean = part.strip()
        if not part_clean:
            continue

        if part_clean.startswith("#"):
            current_section = part_clean.lstrip("#").strip()
            continue

        # If section is small enough, keep as single chunk
        if len(part_clean) <= 900:
            cid = f"{doc_name}_{chunk_counter}"
            chunk_counter += 1
            chunks.append({
                "id": cid,
                "source": f"sample: {doc_name}.md",
                "section": current_section,
                "text": part_clean,
            })
            continue

        # Split with ~900 chars, 120 overlap, sentence boundaries
        start = 0
        while start < len(part_clean):
            end = min(start + 900, len(part_clean))
            if end < len(part_clean):
                # Look for sentence boundary near end
                boundary = -1
                for punct in (". ", ".\n", "? ", "! "):
                    pos = part_clean.rfind(punct, start + 400, end)
                    if pos > boundary:
                        boundary = pos + len(punct)
                if boundary > start:
                    end = boundary

            chunk_text = part_clean[start:end].strip()
            if chunk_text:
                cid = f"{doc_name}_{chunk_counter}"
                chunk_counter += 1
                chunks.append({
                    "id": cid,
                    "source": f"sample: {doc_name}.md",
                    "section": current_section,
                    "text": chunk_text,
                })

            if end >= len(part_clean):
                break
            start = max(end - 120, start + 1)

    return chunks


def build_or_load_knowledge_index() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Index knowledge directory markdown files and cache embeddings."""
    global _INDEX_CHUNKS, _INDEX_VECTORS
    if _INDEX_CHUNKS is not None and _INDEX_VECTORS is not None:
        return _INDEX_CHUNKS, _INDEX_VECTORS

    index_file = settings.DATA_DIR / "rag_index.npy"
    chunks_file = settings.DATA_DIR / "rag_chunks.json"

    kb_dir = settings.KNOWLEDGE_DIR
    doc_paths = list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt"))

    all_chunks: List[Dict[str, Any]] = []
    for dp in sorted(doc_paths):
        all_chunks.extend(chunk_document(dp))

    if not all_chunks:
        # Fallback dummy chunk if folder was empty
        all_chunks = [{
            "id": "sample_0",
            "source": "sample: general_medicine.md",
            "section": "Overview",
            "text": "Clinical decision support knowledge base passage.",
        }]

    model, backend = get_embedding_model()

    if backend == "sentence-transformers":
        texts = [c["text"] for c in all_chunks]
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        texts = [c["text"] for c in all_chunks]
        sparse_vecs = model.fit_transform(texts)
        dense_vecs = sparse_vecs.toarray()
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = dense_vecs / norms

    _INDEX_CHUNKS = all_chunks
    _INDEX_VECTORS = vecs

    # Save to disk
    try:
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(index_file, _INDEX_VECTORS)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(_INDEX_CHUNKS, f)
    except Exception as e:
        logger.warning("Could not persist RAG index to disk: %s", str(e))

    return _INDEX_CHUNKS, _INDEX_VECTORS


def embed_query(query: str) -> np.ndarray:
    """Generate normalized query embedding vector."""
    model, backend = get_embedding_model()
    if backend == "sentence-transformers":
        vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    else:
        sparse_vec = model.transform([query])
        dense = sparse_vec.toarray()[0]
        norm = np.linalg.norm(dense)
        vec = dense / norm if norm > 0 else dense
    return vec


def query_index(query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
    """Perform cosine similarity search against local numpy vector index."""
    chunks, vectors = build_or_load_knowledge_index()
    q_vec = embed_query(query)

    # Cosine similarity is dot product of normalized vectors
    scores = np.dot(vectors, q_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Tuple[Dict[str, Any], float]] = []
    for idx in top_indices:
        results.append((chunks[idx], float(scores[idx])))
    return results


def construct_queries(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
) -> List[str]:
    """
    Construct clinically targeted queries:
    - Imaging findings mapped to diagnostic phrases
    - Lab patterns ordered by urgency
    - Patient presenting notes
    Max 6 queries.
    """
    queries: List[str] = []

    # 1. Imaging findings mapping
    imaging_map = {
        "effusion": "pleural effusion causes and imaging",
        "consolidation": "pulmonary consolidation bacterial pneumonia diagnosis",
        "pneumothorax": "pneumothorax signs symptoms management",
        "cardiomegaly": "cardiomegaly congestive heart failure diagnosis",
        "edema": "pulmonary edema heart failure differential",
        "atelectasis": "atelectasis diagnostic evaluation",
        "infiltrate": "pulmonary infiltrate infection differential",
        "pneumonia": "community acquired pneumonia diagnostic criteria",
    }

    if imaging_report and imaging_report.status == "ok":
        for finding in imaging_report.findings:
            f_lower = finding.finding.lower()
            matched = False
            for k, q_phrase in imaging_map.items():
                if k in f_lower:
                    queries.append(q_phrase)
                    matched = True
                    break
            if not matched:
                queries.append(f"{finding.finding} radiographic differential diagnosis")
            if len(queries) >= 2:
                break

    # 2. Lab patterns (urgent first)
    if lab_report and lab_report.status == "ok":
        sorted_patterns = sorted(
            lab_report.patterns,
            key=lambda p: 0 if p.urgency == "urgent" else (1 if p.urgency == "soon" else 2),
        )
        for pat in sorted_patterns:
            queries.append(f"{pat.name} clinical differential diagnosis workup")
            if len(queries) >= 4:
                break

    # 3. Clinical notes key concepts
    notes_clean = re.sub(r"[\r\n]+", " ", case_input.clinical_text).strip()
    sentences = [s.strip() for s in re.split(r"[.!?]", notes_clean) if len(s.strip()) > 10]
    if sentences:
        queries.append(sentences[0][:120])
    if len(sentences) > 1 and len(queries) < 6:
        queries.append(sentences[1][:120])

    # Fallback default if queries empty
    if not queries:
        queries.append("clinical differential diagnosis and management")

    # Limit to max 6 unique queries
    deduped_queries = []
    seen = set()
    for q in queries:
        if q not in seen:
            deduped_queries.append(q)
            seen.add(q)
        if len(deduped_queries) >= 6:
            break

    return deduped_queries


def run_rag_agent(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
) -> RagReport:
    """Execute complete knowledge base retrieval pipeline."""
    queries = construct_queries(case_input, imaging_report, lab_report)

    all_retrieved: List[Tuple[Dict[str, Any], float, str]] = []
    for q in queries:
        hits = query_index(q, top_k=3)
        for chunk, score in hits:
            all_retrieved.append((chunk, score, q))

    # Sort descending by score
    all_retrieved.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate by chunk ID, keeping top 8
    evidence_list: List[Evidence] = []
    seen_ids = set()
    counter = 1

    for chunk, score, q in all_retrieved:
        if chunk["id"] not in seen_ids:
            seen_ids.add(chunk["id"])
            evidence_list.append(
                Evidence(
                    id=f"S{counter}",
                    source=chunk["source"],
                    section=chunk["section"],
                    passage=chunk["text"],
                    score=round(float(score), 4),
                    query=q,
                )
            )
            counter += 1
            if len(evidence_list) >= 8:
                break

    status_val = "ok" if evidence_list else "empty"
    msg = f"Retrieved {len(evidence_list)} citations across {len(queries)} clinical queries."

    return RagReport(
        status=status_val,
        queries=queries,
        evidence=evidence_list,
        backend=_BACKEND_NAME,
        message=msg,
    )


def run_second_pass_rag(condition: str) -> List[Evidence]:
    """Execute rationale-guided second retrieval for differentials lacking citations."""
    hits = query_index(condition, top_k=2)
    new_evidence: List[Evidence] = []
    for i, (chunk, score) in enumerate(hits, 1):
        clean_tag = re.sub(r"[^a-zA-Z0-9]", "", condition)[:6]
        eid = f"S_{clean_tag}_{i}"
        new_evidence.append(
            Evidence(
                id=eid,
                source=chunk["source"],
                section=chunk["section"],
                passage=chunk["text"],
                score=round(float(score), 4),
                query=condition,
            )
        )
    return new_evidence
