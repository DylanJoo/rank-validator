"""
Ranking metrics for information retrieval evaluation.
"""

import numpy as np
from typing import Dict, List, Union, Optional


def compute_dcg(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Compute Discounted Cumulative Gain (DCG).
    
    Args:
        relevances: List of relevance scores in ranked order
        k: Consider only the top-k items (None for all)
    
    Returns:
        DCG score
    """
    if k is not None:
        relevances = relevances[:k]
    
    if len(relevances) == 0:
        return 0.0
    
    # DCG = rel_1 + sum(rel_i / log2(i+1)) for i = 2..n
    dcg = relevances[0]
    for i, rel in enumerate(relevances[1:], start=2):
        dcg += rel / np.log2(i + 1)
    
    return float(dcg)


def compute_ndcg(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG).
    
    Args:
        relevances: List of relevance scores in ranked order
        k: Consider only the top-k items (None for all)
    
    Returns:
        nDCG score (0 to 1)
    """
    dcg = compute_dcg(relevances, k)
    
    # Ideal DCG: sort relevances in descending order
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def compute_mrr(relevances: List[float]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        relevances: List of binary relevance scores (0 or 1) in ranked order
    
    Returns:
        MRR score (reciprocal rank of first relevant item, or 0 if none)
    """
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def compute_map(all_relevances: List[List[float]]) -> float:
    """
    Compute Mean Average Precision (MAP).
    
    Args:
        all_relevances: List of relevance lists for each query
    
    Returns:
        MAP score
    """
    if len(all_relevances) == 0:
        return 0.0
    
    aps = []
    for relevances in all_relevances:
        if len(relevances) == 0:
            aps.append(0.0)
            continue
            
        num_relevant = sum(1 for r in relevances if r > 0)
        if num_relevant == 0:
            aps.append(0.0)
            continue
        
        # Compute precision at each relevant position
        precision_sum = 0.0
        num_relevant_seen = 0
        
        for i, rel in enumerate(relevances, start=1):
            if rel > 0:
                num_relevant_seen += 1
                precision_at_i = num_relevant_seen / i
                precision_sum += precision_at_i
        
        ap = precision_sum / num_relevant
        aps.append(ap)
    
    return np.mean(aps)


def compute_recall(relevances: List[float], k: Optional[int] = None, 
                   total_relevant: Optional[int] = None) -> float:
    """
    Compute Recall@k.
    
    Args:
        relevances: List of binary relevance scores in ranked order
        k: Consider only top-k items (None for all)
        total_relevant: Total number of relevant items (if None, use sum of relevances)
    
    Returns:
        Recall score (0 to 1)
    """
    if k is not None:
        relevances = relevances[:k]
    
    retrieved_relevant = sum(1 for r in relevances if r > 0)
    
    if total_relevant is None:
        # If not provided, assume all relevant items are in the list
        total_relevant = sum(1 for r in relevances if r > 0)
    
    if total_relevant == 0:
        return 0.0
    
    return retrieved_relevant / total_relevant


def compute_ranking_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int] = [10, 100],
) -> Dict[str, float]:
    """
    Compute ranking metrics for IR evaluation.
    
    Args:
        qrels: Query relevance judgments {qid: {docid: relevance}}
        results: Ranking results {qid: {docid: score}}
        k_values: List of k values for metrics@k
    
    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}
    
    all_ndcg = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    all_mrr = []
    all_map_relevances = []
    
    for qid in qrels:
        if qid not in results:
            # No results for this query
            for k in k_values:
                all_ndcg[k].append(0.0)
                all_recall[k].append(0.0)
            all_mrr.append(0.0)
            all_map_relevances.append([])
            continue
        
        # Get ranked documents and their relevances
        ranked_docs = sorted(
            results[qid].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get relevance scores in ranked order
        relevances = [qrels[qid].get(docid, 0) for docid, _ in ranked_docs]
        total_relevant = sum(1 for rel in qrels[qid].values() if rel > 0)
        
        # Compute metrics at different k values
        for k in k_values:
            all_ndcg[k].append(compute_ndcg(relevances, k))
            all_recall[k].append(compute_recall(relevances, k, total_relevant))
        
        # MRR (no k cutoff)
        all_mrr.append(compute_mrr(relevances))
        
        # MAP relevances
        all_map_relevances.append(relevances)
    
    # Aggregate metrics
    for k in k_values:
        metrics[f"ndcg@{k}"] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0.0
        metrics[f"recall@{k}"] = np.mean(all_recall[k]) if all_recall[k] else 0.0
    
    metrics["mrr"] = np.mean(all_mrr) if all_mrr else 0.0
    metrics["map"] = compute_map(all_map_relevances)
    
    return metrics
