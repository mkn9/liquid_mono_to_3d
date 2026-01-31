#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics for VLM Quality Assessment
Implements BLEU, ROUGE-L, and Semantic Similarity

These metrics provide more nuanced evaluation than simple keyword matching.
"""

import numpy as np
from typing import Dict
import re


def tokenize(text: str) -> list:
    """
    Simple tokenization: lowercase and split on whitespace/punctuation.
    
    Args:
        text: Input text
        
    Returns:
        list: Tokens
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def calculate_bleu_score(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score (simplified single-reference version).
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap.
    Commonly used for machine translation and text generation.
    
    Args:
        reference: Ground truth description
        candidate: Generated description
        
    Returns:
        float: BLEU score (0-1, higher is better)
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if not cand_tokens:
        return 0.0
    
    # Calculate precision for unigrams, bigrams, trigrams, 4-grams
    precisions = []
    for n in range(1, 5):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        cand_ngrams = _get_ngrams(cand_tokens, n)
        
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        
        matches = sum(min(ref_ngrams.count(ng), cand_ngrams.count(ng)) 
                     for ng in set(cand_ngrams))
        precision = matches / len(cand_ngrams)
        precisions.append(precision)
    
    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else \
         np.exp(1 - len(ref_tokens) / len(cand_tokens))
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        bleu = bp * np.exp(np.mean([np.log(p) for p in precisions]))
    
    return float(bleu)


def _get_ngrams(tokens: list, n: int) -> list:
    """Get n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence).
    
    ROUGE-L measures the longest common subsequence between texts.
    Good for measuring sentence-level structural similarity.
    
    Args:
        reference: Ground truth description
        candidate: Generated description
        
    Returns:
        float: ROUGE-L F1 score (0-1, higher is better)
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if not ref_tokens or not cand_tokens:
        return 0.0
    
    # Compute LCS length
    lcs_length = _lcs_length(ref_tokens, cand_tokens)
    
    if lcs_length == 0:
        return 0.0
    
    # Calculate recall and precision
    recall = lcs_length / len(ref_tokens)
    precision = lcs_length / len(cand_tokens)
    
    # F1 score
    if recall + precision == 0:
        return 0.0
    
    f1 = 2 * (recall * precision) / (recall + precision)
    
    return float(f1)


def _lcs_length(seq1: list, seq2: list) -> int:
    """
    Calculate Longest Common Subsequence length using dynamic programming.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using simple word overlap and cosine similarity.
    
    Note: This is a lightweight implementation. For production, consider:
    - sentence-transformers (BERT-based)
    - Universal Sentence Encoder
    - OpenAI embeddings
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score (-1 to 1, higher is better)
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Build vocabulary
    vocab = sorted(set(tokens1 + tokens2))
    
    # Create bag-of-words vectors
    vec1 = np.array([tokens1.count(word) for word in vocab])
    vec2 = np.array([tokens2.count(word) for word in vocab])
    
    # Cosine similarity
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    return float(similarity)


def evaluate_all_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a reference-candidate pair.
    
    Args:
        reference: Ground truth description
        candidate: Generated description
        
    Returns:
        dict: Dictionary with 'bleu', 'rouge_l', 'semantic_similarity' scores
    """
    metrics = {
        "bleu": calculate_bleu_score(reference, candidate),
        "rouge_l": calculate_rouge_l(reference, candidate),
        "semantic_similarity": calculate_semantic_similarity(reference, candidate)
    }
    
    return metrics


def print_metrics_report(metrics: Dict[str, float]):
    """
    Print a formatted report of evaluation metrics.
    
    Args:
        metrics: Dictionary of metric scores
    """
    print("="*60)
    print("Enhanced Evaluation Metrics Report")
    print("="*60)
    print(f"BLEU Score:              {metrics['bleu']:.4f}  (0-1, higher better)")
    print(f"ROUGE-L F1:              {metrics['rouge_l']:.4f}  (0-1, higher better)")
    print(f"Semantic Similarity:     {metrics['semantic_similarity']:.4f}  (-1-1, higher better)")
    print("="*60)
    
    # Overall assessment
    avg_score = (metrics['bleu'] + metrics['rouge_l'] + metrics['semantic_similarity']) / 3
    print(f"\nOverall Average:         {avg_score:.4f}")
    
    if avg_score >= 0.7:
        print("Assessment: ✅ Excellent match")
    elif avg_score >= 0.5:
        print("Assessment: ⚠️ Good match")
    elif avg_score >= 0.3:
        print("Assessment: ⚠️ Partial match")
    else:
        print("Assessment: ❌ Poor match")
    print()


if __name__ == "__main__":
    # Demo
    print("Enhanced Metrics Demo")
    print()
    
    reference = "A straight line trajectory moving primarily in the X direction from (0,0,0) to (1,0,0)"
    candidate = "A linear path along the X axis starting at origin and ending at (1,0,0)"
    
    print("Reference:")
    print(f"  {reference}")
    print()
    print("Candidate:")
    print(f"  {candidate}")
    print()
    
    metrics = evaluate_all_metrics(reference, candidate)
    print_metrics_report(metrics)

