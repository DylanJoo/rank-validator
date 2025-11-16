"""
Unit tests for ranking metrics using ir_measures.
"""

import unittest
from rank_validator.metrics import compute_ranking_metrics


class TestRankingMetrics(unittest.TestCase):
    """Test cases for ranking metrics computation."""
    
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
    
    def test_compute_ranking_metrics_perfect(self):
        """Test metrics with perfect ranking."""
        qrels = {
            "q1": {"d1": 2, "d2": 1, "d3": 0},
        }
        results = {
            "q1": {"d1": 1.0, "d2": 0.5, "d3": 0.1},
        }
        
        metrics = compute_ranking_metrics(qrels, results, k_values=[3])
        
        # With perfect ranking, nDCG should be 1.0
        self.assertAlmostEqual(metrics["ndcg@3"], 1.0, places=2)
    
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
    
    def test_compute_ranking_metrics_empty(self):
        """Test metrics with empty qrels and results."""
        qrels = {}
        results = {}
        
        metrics = compute_ranking_metrics(qrels, results, k_values=[10])
        
        # Should return 0.0 for all metrics
        for metric_name, metric_value in metrics.items():
            self.assertEqual(metric_value, 0.0)
    
    def test_compute_ranking_metrics_graded_relevance(self):
        """Test metrics with graded relevance judgments."""
        qrels = {
            "q1": {"d1": 3, "d2": 2, "d3": 1, "d4": 0},
        }
        results = {
            "q1": {"d1": 1.0, "d2": 0.8, "d3": 0.6, "d4": 0.4},
        }
        
        metrics = compute_ranking_metrics(qrels, results, k_values=[3])
        
        # Graded relevance should work correctly
        self.assertGreater(metrics["ndcg@3"], 0.8)
        self.assertLessEqual(metrics["ndcg@3"], 1.0)


if __name__ == "__main__":
    unittest.main()
