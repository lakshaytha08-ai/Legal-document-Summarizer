import re
import numpy as np
import fitz


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF path or file-like object with PyMuPDF.
    Image-only/scanned PDFs return no selectable text and need OCR.
    """
    doc = None
    try:
        if hasattr(pdf_file, "getvalue"):
            doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
        elif hasattr(pdf_file, "read"):
            data = pdf_file.read()
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open(pdf_file)

        pages = [page.get_text("text").strip() for page in doc]
        return "\n".join(page for page in pages if page).strip()
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from PDF: {exc}") from exc
    finally:
        if doc is not None:
            doc.close()


def preprocess_text(text):
    """Normalize whitespace and punctuation without deleting Unicode letters."""
    if not text:
        return ""
    text = re.sub(r'[\u201c\u201d]', '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_basic_stats(text, chunks):
    """Compute basic document and chunk statistics."""
    words = text.split()
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +')
    sentences = [s for s in sentence_endings.split(text) if len(s.strip()) > 3]
    chunk_lengths = [len(c.split()) for c in chunks]
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "chunk_count": len(chunks),
        "mean_chunk_len": float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
        "median_chunk_len": float(np.median(chunk_lengths)) if chunk_lengths else 0.0,
        "std_chunk_len": float(np.std(chunk_lengths)) if chunk_lengths else 0.0,
    }


def extract_potential_clauses(text):
    """Find likely legal clauses using keyword and sentence-length heuristics."""
    if not text:
        return []
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    keywords = [
        "shall", "agree", "warrant", "liability", "indemnify",
        "terminate", "intellectual property", "confidential", "pay",
        "dispute", "obligat", "remedy", "breach", "governing law"
    ]
    clauses, seen = [], set()
    for sentence in sentences:
        cleaned = sentence.strip()
        lowered = cleaned.lower()
        if 10 < len(cleaned.split()) < 120 and any(k in lowered for k in keywords) and lowered not in seen:
            clauses.append(cleaned)
            seen.add(lowered)
    if len(clauses) < 5:
        candidates = sorted((s.strip() for s in sentences if s.strip()), key=lambda s: len(s.split()), reverse=True)
        for sentence in candidates:
            lowered = sentence.lower()
            if lowered not in seen and 12 < len(sentence.split()) < 100:
                clauses.append(sentence)
                seen.add(lowered)
                if len(clauses) >= 8:
                    break
    return clauses
