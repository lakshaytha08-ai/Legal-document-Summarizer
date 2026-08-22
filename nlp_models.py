import os
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import re
import random
from collections import defaultdict

# Apply GPU dynamically based on availability
device = 0 if torch.cuda.is_available() else -1
pt_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_summarizer():
    """
    Downloads and loads the AllenAI LED-base model for long document summarization.
    """
    model_name = "allenai/led-base-16384"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(pt_device)
    return tokenizer, model

def load_classifier():
    """
    Downloads and loads the BART-large-MNLI zero-shot classification pipeline.
    """
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    return classifier

def chunk_document(text, max_tokens=2000, overlap=200):
    """
    Divides the text into overlapping chunks using a sliding window.
    max_tokens represents the window size in words, and overlap is the overlap size in words.
    """
    words = text.split()
    chunks = []
    step = max_tokens - overlap
    if step <= 0:
        step = max_tokens
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_tokens]
        # Avoid saving tiny trailing chunks
        if len(chunk_words) > 30 or len(chunks) == 0:
            chunks.append(" ".join(chunk_words))
    return chunks

def summarize_chunks_mock(chunks):
    """
    Generates a realistic extractive summary instantly without loading deep learning models.
    """
    if not chunks:
        return "Empty document. No text available to summarize."
    
    sentences = []
    # Key legal words to prioritize in mock summary
    priority_keywords = ["agree", "shall", "payment", "terminate", "confidential", "liability", "dispute", "own"]
    
    for chunk in chunks:
        # Split chunk into sentences
        chunk_sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?]) +', chunk)
        for s in chunk_sents:
            s_clean = s.strip()
            if len(s_clean.split()) > 8:
                sentences.append(s_clean)
                
    # Rank sentences by presence of priority keywords and position
    ranked_sents = []
    seen = set()
    for idx, s in enumerate(sentences):
        if s.lower() not in seen:
            seen.add(s.lower())
            score = 0
            # Early sentences in paragraphs are usually summaries
            if idx < 3:
                score += 3
            # Sentences with legal terms
            for kw in priority_keywords:
                if kw in s.lower():
                    score += 2
            ranked_sents.append((s, score))
            
    # Sort by score and take top 4
    ranked_sents.sort(key=lambda x: x[1], reverse=True)
    summary_sents = [item[0] for item in ranked_sents[:4]]
    
    # Ensure they are in the original reading order
    ordered_sents = []
    for s in sentences:
        if s in summary_sents and s not in ordered_sents:
            ordered_sents.append(s)
            
    summary_text = " ".join(ordered_sents)
    if not summary_text:
        summary_text = "The document details commercial and operational agreements between the parties."
        
    return f"{summary_text}\n\n[Summary generated in Offline Mock Mode.]"

def extract_legal_summary_facts(text, clauses):
    """
    Rule-based extraction layer. This represents the 30% grounded API/rule component of the hybrid summarizer.
    """
    if not text:
        return {
            "overview": "No document text available.",
            "parties": [],
            "key_obligations": [],
            "risk_areas": [],
            "clause_sections": {}
        }

    text_lower = text.lower()
    clauses = clauses or []

    legal_sections = {
        "payment": ["payment", "fee", "fees", "invoice", "price", "amount", "reimburse", "cost"],
        "termination": ["terminate", "termination", "expiry", "expiration", "renew", "cancel", "suspend"],
        "confidentiality": ["confidential", "non-disclosure", "nondisclosure", "secret", "proprietary"],
        "liability": ["liability", "indemnify", "indemnity", "damages", "losses", "warranty"],
        "intellectual_property": ["intellectual property", "patent", "copyright", "trademark", "license"],
        "dispute_resolution": ["dispute", "arbitration", "jurisdiction", "governing law", "venue", "court"],
        "privacy": ["privacy", "data protection", "personal data", "gdpr"],
    }

    section_hits = defaultdict(list)
    for section_name, keywords in legal_sections.items():
        for clause in clauses:
            clause_lower = clause.lower()
            if any(keyword in clause_lower for keyword in keywords):
                section_hits[section_name].append(clause)

    parties = []
    party_patterns = [r"between\s+([^\n]+?)\s+and\s+([^\n]+?)\s+(?:herein|with)", r"the\s+party\s+of\s+the\s+first\s+part\s+.*?the\s+party\s+of\s+the\s+second\s+part"]
    for pattern in party_patterns:
        match = re.search(pattern, text_lower, flags=re.IGNORECASE)
        if match:
            parties.extend([part.strip() for part in match.groups() if part and part.strip()])
    parties = list(dict.fromkeys(parties))[:4]

    key_obligations = []
    obligation_signals = ["shall", "must", "agrees", "undertakes", "will provide", "will deliver", "is responsible"]
    for clause in clauses[:10]:
        clause_lower = clause.lower()
        if any(signal in clause_lower for signal in obligation_signals):
            key_obligations.append(clause)
    key_obligations = key_obligations[:5]

    risk_areas = []
    for section_name, hits in section_hits.items():
        if hits:
            risk_areas.append(f"{section_name.replace('_', ' ').title()} ({len(hits)} clause references)")

    overview = "Legal document reviewed with structured extraction and clause mapping."
    if clauses:
        overview = f"Document contains {len(clauses)} extracted legal clauses across {len(section_hits)} legal sections."

    return {
        "overview": overview,
        "parties": parties,
        "key_obligations": key_obligations,
        "risk_areas": risk_areas[:5],
        "clause_sections": {k: v[:3] for k, v in section_hits.items()}
    }


def summarize_with_hermes(chunks, prompt_override=None):
    """
    Calls a locally-running Hermes/Ollama model for legal summary generation.
    Expects either HERMES_API_URL or OLLAMA_BASE_URL (default http://localhost:11434/api/generate).
    """
    if not chunks:
        return "Empty document. No text available to summarize."

    endpoint = os.getenv("HERMES_API_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/api/generate"
    model_name = os.getenv("HERMES_MODEL") or os.getenv("OLLAMA_MODEL") or "hermes3"

    document_text = "\n\n".join(chunks)
    if prompt_override:
        prompt = prompt_override
    else:
        prompt = (
            "You are a legal contract summarizer. Produce a clean summary of this document with the following structure: "
            "1) Executive overview, 2) key obligations and commercial terms, 3) risk areas, 4) termination/payment/confidentiality notes. "
            "Keep the answer concise but legally meaningful.\n\nDOCUMENT:\n"
            f"{document_text[:30000]}"
        )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    response = requests.post(endpoint, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, dict):
        if "response" in data:
            return data["response"].strip()
        if "content" in data and isinstance(data["content"], str):
            return data["content"].strip()
    if isinstance(data, list):
        bits = []
        for item in data:
            if isinstance(item, dict):
                if "response" in item:
                    bits.append(str(item["response"]))
                elif "text" in item:
                    bits.append(str(item["text"]))
        if bits:
            return "".join(bits).strip()

    raise ValueError("Hermes local response was not in the expected format.")


def summarize_document_hybrid(text, chunks, clauses, tokenizer=None, model=None):
    """
    Hybrid legal summarization architecture:
    - 30% API/rule-grounded extraction from sections and clause facts
    - 70% LLM narrative synthesis from the extracted legal facts and full text
    """
    if not text:
        return "Empty document. No text available to summarize."

    facts = extract_legal_summary_facts(text, clauses)
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    contract_hint = "\n\n".join(chunks[:8]) if chunks else text[:5000]

    prompt = (
        "You are a legal contract summarizer. Use the extracted legal facts below as the grounded source of truth. "
        "Apply a 3:7 architecture where the extracted facts cover 30% of the reasoning and the narrative summary covers 70%. "
        "Write a concise but defensible summary in plain English for legal review.\n\n"
        "Required output structure:\n"
        "1. Executive Overview\n"
        "2. Key Obligations\n"
        "3. Payment and Commercial Terms\n"
        "4. Risks and Red Flags\n"
        "5. Termination and Compliance Notes\n\n"
        "EXTRACTED FACTS:\n"
        f"{facts_json}\n\n"
        "FULL DOCUMENT CONTEXT:\n"
        f"{contract_hint[:20000]}"
    )

    hermes_api = os.getenv("HERMES_API_URL") or os.getenv("OLLAMA_BASE_URL")
    hermes_model = os.getenv("HERMES_MODEL") or os.getenv("OLLAMA_MODEL")
    if hermes_api or hermes_model:
        try:
            return summarize_with_hermes(chunks, prompt_override=prompt)
        except Exception:
            pass

    if tokenizer is not None and model is not None:
        try:
            inputs = tokenizer(contract_hint[:15000], return_tensors="pt", max_length=8192, truncation=True).to(pt_device)
            global_attention_mask = torch.zeros_like(inputs.input_ids)
            global_attention_mask[:, 0] = 1
            summary_ids = model.generate(inputs.input_ids, global_attention_mask=global_attention_mask, num_beams=3, max_length=256)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            if summary.strip():
                return summary.strip()
        except Exception:
            pass

    return summarize_chunks_mock(chunks)


def summarize_chunks(chunks, tokenizer, model, use_mock=False):
    """
    Generates summaries for document chunks.
    If use_mock is True, runs the heuristic offline summarizer.
    If a local Hermes/Ollama endpoint is configured, it is used automatically.
    """
    if use_mock:
        return summarize_chunks_mock(chunks)

    hermes_api = os.getenv("HERMES_API_URL") or os.getenv("OLLAMA_BASE_URL")
    hermes_model = os.getenv("HERMES_MODEL") or os.getenv("OLLAMA_MODEL")
    if hermes_api or hermes_model:
        try:
            return summarize_with_hermes(chunks)
        except Exception:
            pass

    summaries = []
    for chunk in chunks:
        # Encode the chunk
        inputs = tokenizer(chunk, return_tensors="pt", max_length=8192, truncation=True).to(pt_device)
        global_attention_mask = torch.zeros_like(inputs.input_ids)
        # Put global attention on <s> token for LED base requirement
        global_attention_mask[:, 0] = 1 

        summary_ids = model.generate(inputs.input_ids, global_attention_mask=global_attention_mask, num_beams=3, max_length=256)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return " \n\n ".join(summaries)

def classify_clauses_mock(clauses):
    """
    Classifies clauses using keyword heuristics instantly.
    """
    categories = [
        "Confidentiality", 
        "Payment", 
        "Termination", 
        "Liability", 
        "Intellectual Property", 
        "Dispute Resolution"
    ]
    
    # Keywords mapped to categories
    keywords = {
        "Confidentiality": ["confidential", "nondisclose", "secret", "private", "disclosure", "proprietary"],
        "Payment": ["pay", "fee", "invoice", "price", "charge", "cost", "sum", "reimburse", "amount", "cash", "dollar"],
        "Termination": ["terminate", "termination", "expire", "expiration", "cancel", "period", "duration", "renew"],
        "Liability": ["liable", "liability", "damage", "indemnity", "indemnify", "consequential", "punitive", "negligence"],
        "Intellectual Property": ["intellectual", "patent", "copyright", "trademark", "license", "ownership", "proprietary", "software", "author"],
        "Dispute Resolution": ["dispute", "arbitration", "arbitrate", "court", "jurisdiction", "governing law", "venue", "lawsuit", "aaa"]
    }
    
    results = []
    # Set seed for reproducible offline statistics and debugging
    random.seed(42)
    
    for clause in clauses:
        clause_lower = clause.lower()
        scores = {cat: 0.0 for cat in categories}
        
        # Word counts for each keyword
        for cat, keys in keywords.items():
            for key in keys:
                if key in clause_lower:
                    scores[cat] += 1.5
                    
        best_cat = max(scores, key=scores.get)
        max_score = scores[best_cat]
        
        if max_score == 0:
            # Default fallback if no keywords matched
            best_cat = random.choice(categories)
            confidence = float(random.uniform(0.45, 0.65))
        else:
            # Normalize confidence score
            confidence = float(min(0.99, 0.70 + (max_score * 0.06) + random.uniform(-0.02, 0.02)))
            
        results.append({
            "clause": clause,
            "category": best_cat,
            "confidence": confidence,
            "uncertainty": float(1.0 - confidence)
        })
        
    return results

def classify_clauses(clauses, classifier, use_mock=False):
    """
    Classifies extracted legal clauses into predefined categories using zero-shot classification.
    """
    if use_mock:
        return classify_clauses_mock(clauses)
        
    categories = [
        "Confidentiality", 
        "Payment", 
        "Termination", 
        "Liability", 
        "Intellectual Property", 
        "Dispute Resolution"
    ]
    results = []
    for clause in clauses:
        out = classifier(clause, candidate_labels=categories)
        best_label = out['labels'][0]
        confidence = out['scores'][0]
        results.append({
            "clause": clause,
            "category": best_label,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence
        })
    return results

def calculate_classification_stats(classifications):
    """
    Computes statistical indicators (mean, median, std) for prediction confidence and uncertainty.
    """
    if not classifications:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    confidences = [c["confidence"] for c in classifications]
    uncertainties = [c["uncertainty"] for c in classifications]
    
    return (
        float(np.mean(confidences)), 
        float(np.median(confidences)), 
        float(np.std(confidences)),
        float(np.mean(uncertainties)),
        float(np.median(uncertainties)),
        float(np.std(uncertainties))
    )
