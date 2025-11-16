"""
Dataset loader for IR testing sets from HuggingFace.
"""

from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import json


class IRDatasetLoader:
    """
    Loader for Information Retrieval datasets from HuggingFace.
    
    Expects datasets to have:
    - queries: Dataset with columns [qid, text]
    - corpus: Dataset with columns [docid, text]
    - qrels: Query relevance judgments {qid: {docid: relevance}}
    - results: Top-k BM25 retrieved results {qid: {docid: score}}
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            split: Dataset split to load (e.g., 'test', 'dev')
            cache_dir: Directory to cache the dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        
        self._queries = None
        self._corpus = None
        self._qrels = None
        self._bm25_results = None
    
    def load_queries(self) -> Dict[str, str]:
        """
        Load queries from the dataset.
        
        Returns:
            Dictionary mapping query IDs to query text
        """
        if self._queries is None:
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    "queries",
                    split=self.split,
                    cache_dir=self.cache_dir,
                )
                self._queries = {
                    str(item["qid"]): item["text"]
                    for item in dataset
                }
            except Exception as e:
                print(f"Warning: Could not load queries: {e}")
                self._queries = {}
        
        return self._queries
    
    def load_corpus(self) -> Dict[str, str]:
        """
        Load corpus from the dataset.
        
        Returns:
            Dictionary mapping document IDs to document text
        """
        if self._corpus is None:
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    "corpus",
                    split=self.split,
                    cache_dir=self.cache_dir,
                )
                self._corpus = {
                    str(item["docid"]): item["text"]
                    for item in dataset
                }
            except Exception as e:
                print(f"Warning: Could not load corpus: {e}")
                self._corpus = {}
        
        return self._corpus
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """
        Load query relevance judgments.
        
        Returns:
            Dictionary {qid: {docid: relevance}}
        """
        if self._qrels is None:
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    "qrels",
                    split=self.split,
                    cache_dir=self.cache_dir,
                )
                
                self._qrels = {}
                for item in dataset:
                    qid = str(item["qid"])
                    docid = str(item["docid"])
                    relevance = int(item["relevance"])
                    
                    if qid not in self._qrels:
                        self._qrels[qid] = {}
                    self._qrels[qid][docid] = relevance
                    
            except Exception as e:
                print(f"Warning: Could not load qrels: {e}")
                self._qrels = {}
        
        return self._qrels
    
    def load_bm25_results(self, top_k: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Load BM25 retrieval results.
        
        Args:
            top_k: Number of top results to load per query
        
        Returns:
            Dictionary {qid: {docid: score}}
        """
        if self._bm25_results is None:
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    "bm25_results",
                    split=self.split,
                    cache_dir=self.cache_dir,
                )
                
                self._bm25_results = {}
                for item in dataset:
                    qid = str(item["qid"])
                    
                    # Handle different possible formats
                    if "docids" in item and "scores" in item:
                        # List format
                        docids = item["docids"][:top_k]
                        scores = item["scores"][:top_k]
                        
                        self._bm25_results[qid] = {
                            str(docid): float(score)
                            for docid, score in zip(docids, scores)
                        }
                    elif "results" in item:
                        # Dict format
                        results = item["results"]
                        if isinstance(results, str):
                            results = json.loads(results)
                        
                        # Take top_k results
                        sorted_results = sorted(
                            results.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:top_k]
                        
                        self._bm25_results[qid] = {
                            str(docid): float(score)
                            for docid, score in sorted_results
                        }
                    
            except Exception as e:
                print(f"Warning: Could not load BM25 results: {e}")
                self._bm25_results = {}
        
        return self._bm25_results
    
    def load_all(self, top_k: int = 100) -> Tuple[
        Dict[str, str],
        Dict[str, str],
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, float]]
    ]:
        """
        Load all components of the dataset.
        
        Args:
            top_k: Number of top BM25 results to load
        
        Returns:
            Tuple of (queries, corpus, qrels, bm25_results)
        """
        queries = self.load_queries()
        corpus = self.load_corpus()
        qrels = self.load_qrels()
        bm25_results = self.load_bm25_results(top_k)
        
        return queries, corpus, qrels, bm25_results
