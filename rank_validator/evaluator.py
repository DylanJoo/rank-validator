"""
Rank evaluator for computing ranking metrics using pre-retrieved results.
"""

from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .dataset_loader import IRDatasetLoader
from .metrics import compute_ranking_metrics


class RankEvaluator:
    """
    Evaluator for ranking models using BM25 pre-retrieved results.
    
    This evaluator:
    1. Loads IR test set and BM25 top-k results
    2. Re-ranks the BM25 results using the model
    3. Computes ranking metrics (nDCG, MRR, MAP, Recall)
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        top_k: int = 100,
        k_values: List[int] = [10, 100],
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the rank evaluator.
        
        Args:
            dataset_name: Name of the HuggingFace IR dataset
            split: Dataset split to use
            top_k: Number of top BM25 results to re-rank
            k_values: List of k values for metrics@k
            cache_dir: Directory to cache datasets
            batch_size: Batch size for model inference
        """
        self.dataset_name = dataset_name
        self.split = split
        self.top_k = top_k
        self.k_values = k_values
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Load dataset
        self.loader = IRDatasetLoader(dataset_name, split, cache_dir)
        self.queries, self.corpus, self.qrels, self.bm25_results = self.loader.load_all(top_k)
        
        print(f"Loaded {len(self.queries)} queries, {len(self.corpus)} documents")
        print(f"Loaded {len(self.qrels)} query relevance judgments")
        print(f"Loaded BM25 results for {len(self.bm25_results)} queries")
    
    def create_rerank_pairs(self) -> List[Dict[str, str]]:
        """
        Create query-document pairs for re-ranking.
        
        Returns:
            List of dictionaries with 'qid', 'docid', 'query', 'document'
        """
        pairs = []
        
        for qid in self.bm25_results:
            if qid not in self.queries:
                continue
            
            query_text = self.queries[qid]
            
            for docid in self.bm25_results[qid]:
                if docid in self.corpus:
                    pairs.append({
                        "qid": qid,
                        "docid": docid,
                        "query": query_text,
                        "document": self.corpus[docid],
                    })
        
        return pairs
    
    def evaluate(
        self,
        model,
        tokenizer=None,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Evaluate a ranking model.
        
        Args:
            model: The ranking model (should output relevance scores)
            tokenizer: Tokenizer for the model (if needed)
            device: Device to run inference on
        
        Returns:
            Dictionary of metric names to scores
        """
        model.eval()
        model.to(device)
        
        # Create re-ranking pairs
        pairs = self.create_rerank_pairs()
        
        if len(pairs) == 0:
            print("Warning: No query-document pairs to evaluate")
            return {f"{metric}@{k}": 0.0 for metric in ["ndcg", "recall"] for k in self.k_values}
        
        # Get model scores
        rerank_results = self._score_pairs(model, tokenizer, pairs, device)
        
        # Compute metrics
        metrics = compute_ranking_metrics(self.qrels, rerank_results, self.k_values)
        
        return metrics
    
    def _score_pairs(
        self,
        model,
        tokenizer,
        pairs: List[Dict[str, str]],
        device: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Score query-document pairs using the model.
        
        Args:
            model: The ranking model
            tokenizer: Tokenizer for the model
            pairs: List of query-document pairs
            device: Device to run inference on
        
        Returns:
            Dictionary {qid: {docid: score}}
        """
        results = {}
        
        # Batch process pairs
        for i in tqdm(range(0, len(pairs), self.batch_size), desc="Scoring"):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Prepare inputs
            queries = [p["query"] for p in batch_pairs]
            documents = [p["document"] for p in batch_pairs]
            
            # Tokenize
            if tokenizer is not None:
                inputs = tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                # Assume model handles raw text
                inputs = {"queries": queries, "documents": documents}
            
            # Get scores
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Handle different output formats
                if isinstance(outputs, torch.Tensor):
                    scores = outputs.squeeze(-1)
                elif hasattr(outputs, "logits"):
                    scores = outputs.logits.squeeze(-1)
                else:
                    raise ValueError(f"Unsupported model output type: {type(outputs)}")
                
                scores = scores.cpu().numpy()
            
            # Organize results by query
            for j, pair in enumerate(batch_pairs):
                qid = pair["qid"]
                docid = pair["docid"]
                score = float(scores[j])
                
                if qid not in results:
                    results[qid] = {}
                results[qid][docid] = score
        
        return results
    
    def evaluate_with_scores(
        self,
        scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Evaluate pre-computed scores.
        
        Args:
            scores: Dictionary {qid: {docid: score}}
        
        Returns:
            Dictionary of metric names to scores
        """
        metrics = compute_ranking_metrics(self.qrels, scores, self.k_values)
        return metrics
