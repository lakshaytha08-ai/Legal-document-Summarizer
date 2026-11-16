import re
import numpy as np
import fitz


def extract_text_from_pdf(pdf_file, ocr_language="eng", ocr_dpi=200):
    """
    Extract selectable PDF text and fall back to Tesseract OCR for scanned pages.
    Accepts a file path or a file-like object.
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

        extracted_pages = []
        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()

            # OCR only pages that contain no selectable text.
            if not page_text:
                try:
                    text_page = page.get_textpage_ocr(
                        language=ocr_language,
                        dpi=ocr_dpi,
                        full=True,
                    )
                    page_text = page.get_text("text", textpage=text_page).strip()
                except Exception as ocr_error:
                    raise RuntimeError(
                        f"OCR failed on PDF page {page_number}. "
                        "Make sure Tesseract OCR and the requested language data "
                        f"are installed. Details: {ocr_error}"
                    ) from ocr_error

            if page_text:
                extracted_pages.append(page_text)

        return "\\n".join(extracted_pages).strip()
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
