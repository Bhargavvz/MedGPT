"""
VQA Metrics Module
==================
Evaluation metrics for Visual Question Answering.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import numpy as np


def compute_vqa_metrics(
    predictions: List,
    references: List,
    answer_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive VQA metrics.
    
    Args:
        predictions: Model predictions
        references: Ground truth answers
        answer_types: Optional answer type labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Exact match accuracy
    metrics['accuracy'] = compute_accuracy(predictions, references)
    metrics['exact_match'] = compute_exact_match(predictions, references)
    
    # F1 score
    metrics['f1'] = compute_f1(predictions, references)
    
    # BLEU scores
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)
    
    # ROUGE-L
    metrics['rouge_l'] = compute_rouge_l(predictions, references)
    
    # By answer type (if provided)
    if answer_types:
        type_metrics = compute_metrics_by_type(predictions, references, answer_types)
        metrics.update(type_metrics)
    
    return metrics


def compute_accuracy(
    predictions: List,
    references: List,
) -> float:
    """
    Compute accuracy (VQA-style with multiple correct answers).
    
    For VQA, accuracy is min(#humans agreeing / 3, 1).
    Here we use simple match for single reference.
    """
    if not predictions or not references:
        return 0.0
    
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_answer(str(pred))
        ref_norm = normalize_answer(str(ref))
        
        if pred_norm == ref_norm:
            correct += 1
    
    return correct / total


def compute_exact_match(
    predictions: List,
    references: List,
) -> float:
    """Compute exact match score."""
    if not predictions or not references:
        return 0.0
    
    exact = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        if str(pred).strip().lower() == str(ref).strip().lower():
            exact += 1
    
    return exact / total


def compute_f1(
    predictions: List,
    references: List,
) -> float:
    """Compute token-level F1 score."""
    if not predictions or not references:
        return 0.0
    
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(str(pred)).split()
        ref_tokens = normalize_answer(str(ref)).split()
        
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            f1_scores.append(0.0)
            continue
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


def compute_bleu(
    predictions: List,
    references: List,
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU scores (1-4 grams).
    
    Simplified implementation without smoothing.
    """
    bleu_scores = {f'bleu_{n}': [] for n in range(1, max_n + 1)}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(str(pred)).split()
        ref_tokens = normalize_answer(str(ref)).split()
        
        for n in range(1, max_n + 1):
            if len(pred_tokens) < n or len(ref_tokens) < n:
                bleu_scores[f'bleu_{n}'].append(0.0)
                continue
            
            # Get n-grams
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            # Count matches
            common = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            if total == 0:
                bleu_scores[f'bleu_{n}'].append(0.0)
            else:
                bleu_scores[f'bleu_{n}'].append(common / total)
    
    # Average
    return {k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()}


def compute_rouge_l(
    predictions: List,
    references: List,
) -> float:
    """
    Compute ROUGE-L score (longest common subsequence).
    """
    if not predictions or not references:
        return 0.0
    
    rouge_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_answer(str(pred)).split()
        ref_tokens = normalize_answer(str(ref)).split()
        
        if not pred_tokens or not ref_tokens:
            rouge_scores.append(0.0)
            continue
        
        # LCS length
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        # Precision and recall
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        
        if precision + recall == 0:
            rouge_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            rouge_scores.append(f1)
    
    return np.mean(rouge_scores)


def compute_metrics_by_type(
    predictions: List,
    references: List,
    answer_types: List[str],
) -> Dict[str, float]:
    """Compute metrics grouped by answer type."""
    type_groups = {}
    
    for pred, ref, atype in zip(predictions, references, answer_types):
        if atype not in type_groups:
            type_groups[atype] = {'preds': [], 'refs': []}
        type_groups[atype]['preds'].append(pred)
        type_groups[atype]['refs'].append(ref)
    
    metrics = {}
    for atype, data in type_groups.items():
        type_acc = compute_accuracy(data['preds'], data['refs'])
        metrics[f'accuracy_{atype}'] = type_acc
    
    return metrics


# Helper functions

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    s = s.lower()
    
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)
    
    # Remove extra whitespace
    s = ' '.join(s.split())
    
    return s


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-grams from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    return Counter(ngrams)


def lcs_length(seq1: List, seq2: List) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


class VQAEvaluator:
    """
    Comprehensive VQA evaluator class.
    """
    
    def __init__(
        self,
        tokenizer=None,
        answer_vocab: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            tokenizer: Tokenizer for decoding
            answer_vocab: Answer vocabulary mapping
        """
        self.tokenizer = tokenizer
        self.answer_vocab = answer_vocab or {}
        
        # Storage for results
        self.predictions = []
        self.references = []
        self.answer_types = []
    
    def add_prediction(
        self,
        prediction: Union[str, int, List[int]],
        reference: Union[str, int],
        answer_type: Optional[str] = None,
    ):
        """Add a prediction-reference pair."""
        # Decode if needed
        if isinstance(prediction, list) and self.tokenizer:
            prediction = self.tokenizer.decode(prediction, skip_special_tokens=True)
        elif isinstance(prediction, int) and self.answer_vocab:
            prediction = self.answer_vocab.get(prediction, str(prediction))
        
        if isinstance(reference, int) and self.answer_vocab:
            reference = self.answer_vocab.get(reference, str(reference))
        
        self.predictions.append(prediction)
        self.references.append(reference)
        if answer_type:
            self.answer_types.append(answer_type)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        return compute_vqa_metrics(
            predictions=self.predictions,
            references=self.references,
            answer_types=self.answer_types if self.answer_types else None,
        )
    
    def reset(self):
        """Reset stored predictions."""
        self.predictions = []
        self.references = []
        self.answer_types = []
    
    def summary(self) -> str:
        """Get metrics summary string."""
        metrics = self.compute()
        
        lines = ["VQA Evaluation Results", "=" * 30]
        for name, value in sorted(metrics.items()):
            lines.append(f"{name}: {value:.4f}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics
    print("Testing VQA Metrics...")
    
    predictions = [
        "yes",
        "pneumonia",
        "the left lung",
        "no abnormality detected",
    ]
    
    references = [
        "yes",
        "pneumonia",
        "left lung",
        "no abnormality",
    ]
    
    answer_types = ["yes_no", "diagnosis", "location", "finding"]
    
    metrics = compute_vqa_metrics(predictions, references, answer_types)
    
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nVQA Metrics test passed!")
