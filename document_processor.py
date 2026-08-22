import PyPDF2
import re
import numpy as np

def extract_text_from_pdf(pdf_file):
    """
    Extracts raw text from a PDF file stream.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def preprocess_text(text):
    """
    Cleans raw text by consolidating whitespace, stripping unicode garbage,
    and removing common OCR artifact stutters.
    """
    if not text:
        return ""
    # Replace non-ascii quotes and hyphens with standard ones
    text = re.sub(r'[\u201c\u201d]', '"', text)
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u2013\u2014]', "-", text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove non-ascii characters to ensure NLP models handle it clean
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def calculate_basic_stats(text, chunks):
    """
    Computes statistical indicators for the processed document.
    """
    words = text.split()
    # Robust sentence split avoiding splitting on common legal abbreviations
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    sentences = [s for s in sentences if len(s.strip()) > 3]
    
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
    """
    Identifies sentences or paragraphs that are likely to contain important legal clauses
    using keyword matching and length filters.
    """
    if not text:
        return []
        
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    
    # Predefined legal indicators
    keywords = [
        'shall', 'agree', 'warrant', 'liability', 'indemnify',
        'terminate', 'intellectual property', 'confidential', 
        'pay', 'dispute', 'obligat', 'remedy', 'breach', 'governing law'
    ]
    
    clauses = []
    seen = set()
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        word_count = len(s_clean.split())
        # Focus on readable, descriptive clauses (usually between 10 and 120 words)
        if 10 < word_count < 120:
            s_lower = s_clean.lower()
            if any(k in s_lower for k in keywords):
                if s_lower not in seen:
                    seen.add(s_lower)
                    clauses.append(s_clean)
    
    # Fallback if too few clauses are extracted: grab longest sentences
    if len(clauses) < 5:
        sorted_sents = sorted([s.strip() for s in sentences if s.strip()], key=lambda x: len(x.split()), reverse=True)
        for s in sorted_sents:
            if s.lower() not in seen and 12 < len(s.split()) < 100:
                clauses.append(s)
                seen.add(s.lower())
                if len(clauses) >= 8:
                    break
                    
    return clauses
