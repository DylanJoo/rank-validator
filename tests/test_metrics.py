"""
Unit tests for ranking metrics.
"""

import unittest
from rank_validator.metrics import (
    compute_dcg,
    compute_ndcg,
    compute_mrr,
    compute_map,
    compute_recall,
    compute_ranking_metrics,
)


class TestRankingMetrics(unittest.TestCase):
    """Test cases for ranking metrics computation."""
    
    def test_dcg_basic(self):
        """Test basic DCG computation."""
        relevances = [3, 2, 3, 0, 1, 2]
        dcg = compute_dcg(relevances)
        self.assertGreater(dcg, 0)
        
    def test_dcg_empty(self):
        """Test DCG with empty relevances."""
        relevances = []
        dcg = compute_dcg(relevances)
        self.assertEqual(dcg, 0.0)
    
    def test_dcg_with_k(self):
        """Test DCG with cutoff k."""
        relevances = [3, 2, 3, 0, 1, 2]
        dcg_full = compute_dcg(relevances)
        dcg_k3 = compute_dcg(relevances, k=3)
        self.assertLess(dcg_k3, dcg_full)
    
    def test_ndcg_perfect(self):
        """Test nDCG with perfect ranking."""
        relevances = [3, 2, 1, 0]
        ndcg = compute_ndcg(relevances)
        self.assertAlmostEqual(ndcg, 1.0)
    
    def test_ndcg_worst(self):
        """Test nDCG with worst ranking."""
        relevances = [0, 1, 2, 3]
        ndcg = compute_ndcg(relevances)
        self.assertLess(ndcg, 0.8)
    
    def test_ndcg_with_k(self):
        """Test nDCG with cutoff k."""
        relevances = [3, 2, 1, 0]
        ndcg = compute_ndcg(relevances, k=2)
        self.assertGreater(ndcg, 0)
        self.assertLessEqual(ndcg, 1.0)
    
    def test_mrr_basic(self):
        """Test basic MRR computation."""
        relevances = [0, 0, 1, 0]
        mrr = compute_mrr(relevances)
        self.assertAlmostEqual(mrr, 1.0 / 3)
    
    def test_mrr_first_position(self):
        """Test MRR with relevant item at first position."""
        relevances = [1, 0, 0]
        mrr = compute_mrr(relevances)
        self.assertAlmostEqual(mrr, 1.0)
    
    def test_mrr_no_relevant(self):
        """Test MRR with no relevant items."""
        relevances = [0, 0, 0]
        mrr = compute_mrr(relevances)
        self.assertEqual(mrr, 0.0)
    
    def test_recall_basic(self):
        """Test basic recall computation."""
        relevances = [1, 0, 1, 0, 1]
        recall = compute_recall(relevances, k=3, total_relevant=3)
        self.assertAlmostEqual(recall, 2.0 / 3)
    
    def test_recall_all_retrieved(self):
        """Test recall when all relevant items are retrieved."""
        relevances = [1, 1, 0, 0]
        recall = compute_recall(relevances, k=4, total_relevant=2)
        self.assertAlmostEqual(recall, 1.0)
    
    def test_map_basic(self):
        """Test basic MAP computation."""
        all_relevances = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ]
        map_score = compute_map(all_relevances)
        self.assertGreater(map_score, 0)
        self.assertLessEqual(map_score, 1.0)
    
    def test_map_empty(self):
        """Test MAP with empty queries."""
        all_relevances = []
        map_score = compute_map(all_relevances)
        self.assertEqual(map_score, 0.0)
    
    def test_compute_ranking_metrics_basic(self):
        """Test computing all ranking metrics."""
        qrels = {
            "q1": {"d1": 1, "d2": 0, "d3": 1},
            "q2": {"d1": 1, "d2": 1, "d3": 0},
        }
        results = {
            "q1": {"d1": 0.9, "d3": 0.8, "d2": 0.1},
            "q2": {"d1": 0.9, "d2": 0.8, "d3": 0.1},
        }
        
        metrics = compute_ranking_metrics(qrels, results, k_values=[2, 3])
        
        # Check that all expected metrics are present
        self.assertIn("ndcg@2", metrics)
        self.assertIn("ndcg@3", metrics)
        self.assertIn("recall@2", metrics)
        self.assertIn("recall@3", metrics)
        self.assertIn("mrr", metrics)
        self.assertIn("map", metrics)
        
        # Check metric values are in valid range
        for metric_name, metric_value in metrics.items():
            self.assertGreaterEqual(metric_value, 0.0)
            self.assertLessEqual(metric_value, 1.0)
    
    def test_compute_ranking_metrics_missing_queries(self):
        """Test metrics computation with missing queries in results."""
        qrels = {
            "q1": {"d1": 1, "d2": 0},
            "q2": {"d1": 1, "d2": 0},
        }
        results = {
            "q1": {"d1": 0.9, "d2": 0.1},
            # q2 is missing
        }
        
        metrics = compute_ranking_metrics(qrels, results, k_values=[2])
        
        # Should handle missing queries gracefully
        self.assertIn("ndcg@2", metrics)
        self.assertGreaterEqual(metrics["ndcg@2"], 0.0)


if __name__ == "__main__":
    unittest.main()
