"""
Ranking metrics for information retrieval evaluation using ir_measures.
"""

from typing import Dict, List
import ir_measures
from ir_measures import nDCG, RR, AP, R


def compute_ranking_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int] = [10, 100],
) -> Dict[str, float]:
    """
    Compute ranking metrics for IR evaluation using ir_measures.
    
    Args:
        qrels: Query relevance judgments {qid: {docid: relevance}}
        results: Ranking results {qid: {docid: score}}
        k_values: List of k values for metrics@k
    
    Returns:
        Dictionary of metric names to scores
    """
    # Handle empty case
    if not qrels or not results:
        output = {}
        for k in k_values:
            output[f"ndcg@{k}"] = 0.0
            output[f"recall@{k}"] = 0.0
        output["mrr"] = 0.0
        output["map"] = 0.0
        return output
    
    # Convert qrels to ir_measures format
    qrels_list = []
    for qid, doc_rels in qrels.items():
        for docid, rel in doc_rels.items():
            qrels_list.append(ir_measures.Qrel(qid, docid, rel))
    
    # Convert results to ir_measures format
    run_list = []
    for qid, doc_scores in results.items():
        for docid, score in doc_scores.items():
            run_list.append(ir_measures.ScoredDoc(qid, docid, score))
    
    # Define metrics to compute
    metrics_to_compute = []
    for k in k_values:
        metrics_to_compute.append(nDCG @ k)
        metrics_to_compute.append(R @ k)
    
    # Add MRR and MAP
    metrics_to_compute.append(RR)
    metrics_to_compute.append(AP)
    
    # Compute metrics
    results_dict = ir_measures.calc_aggregate(metrics_to_compute, qrels_list, run_list)
    
    # Format output
    output = {}
    for k in k_values:
        output[f"ndcg@{k}"] = results_dict.get(nDCG @ k, 0.0)
        output[f"recall@{k}"] = results_dict.get(R @ k, 0.0)
    
    output["mrr"] = results_dict.get(RR, 0.0)
    output["map"] = results_dict.get(AP, 0.0)
    
    return output
