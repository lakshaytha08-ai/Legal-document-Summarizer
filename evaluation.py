from rouge_score import rouge_scorer
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import re

def calculate_offline_semantic_similarity(text1, text2):
    """
    Computes a mock/fallback semantic similarity between two texts using token Jaccard similarity
    and 3-gram character overlap. Used when offline or BERTScore cannot download models.
    """
    def get_tokens(text):
        return set(re.findall(r'\w+', text.lower()))
        
    tokens1 = get_tokens(text1)
    tokens2 = get_tokens(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
        
    # Token-level Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # 3-Gram character overlap
    def get_3grams(text):
        text_clean = re.sub(r'\s+', ' ', text.lower())
        return set(text_clean[i:i+3] for i in range(len(text_clean)-2))
        
    grams1 = get_3grams(text1)
    grams2 = get_3grams(text2)
    
    char_jaccard = len(grams1.intersection(grams2)) / len(grams1.union(grams2)) if (grams1 or grams2) else 0.0
    
    # Combined score mapped to a realistic BERTScore range (usually 0.5 to 0.95 for legal text)
    combined = 0.5 * jaccard + 0.5 * char_jaccard
    bertscore_est = 0.45 + (combined * 0.55)
    return float(min(1.0, max(0.0, bertscore_est)))

def evaluate_summary(generated_summary, reference_summary):
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore for the generated summary.
    """
    if not generated_summary or not reference_summary:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bertscore_f1": 0.0,
            "is_bertscore_real": False
        }
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    # Optional BERTScore compute
    is_real = True
    try:
        # Load BERTScore evaluator
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(
            predictions=[generated_summary], 
            references=[reference_summary], 
            lang="en",
            model_type="distilbert-base-uncased" # Lightweight model to avoid massive downloads
        )
        b_score = float(results["f1"][0])
    except Exception:
        # Fallback to local heuristic estimation if offline
        b_score = calculate_offline_semantic_similarity(generated_summary, reference_summary)
        is_real = False
        
    return {
        "rouge1": float(scores['rouge1'].fmeasure),
        "rouge2": float(scores['rouge2'].fmeasure),
        "rougeL": float(scores['rougeL'].fmeasure),
        "bertscore_f1": b_score,
        "is_bertscore_real": is_real
    }

def evaluate_classification(true_labels, predicted_labels):
    """
    Computes average Precision, Recall, and F1-Score, and returns per-category metrics.
    """
    if not true_labels or not predicted_labels:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "class_report": {}
        }
        
    # Ensure consistent lengths to avoid ValueError in scikit-learn
    if len(true_labels) != len(predicted_labels):
        min_len = min(len(true_labels), len(predicted_labels))
        true_labels = true_labels[:min_len]
        predicted_labels = predicted_labels[:min_len]
        
    precision = float(precision_score(true_labels, predicted_labels, average='weighted', zero_division=0))
    recall = float(recall_score(true_labels, predicted_labels, average='weighted', zero_division=0))
    f1 = float(f1_score(true_labels, predicted_labels, average='weighted', zero_division=0))
    
    # Compute detailed classification report
    try:
        report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
        # Clean up report to only include categories, not averages
        class_report = {}
        for key, val in report.items():
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                class_report[key] = {
                    "precision": float(val["precision"]),
                    "recall": float(val["recall"]),
                    "f1-score": float(val["f1-score"]),
                    "support": int(val["support"])
                }
    except Exception:
        class_report = {}
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_report": class_report
    }
