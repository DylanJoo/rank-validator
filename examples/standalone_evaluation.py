"""
Example script showing how to use RankEvaluator standalone (without Trainer).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_validator import RankEvaluator


def main():
    # Load your trained model
    model_name = "your-username/your-trained-model"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create the evaluator
    evaluator = RankEvaluator(
        dataset_name="your-username/ir-test-dataset",  # Replace with your IR dataset
        split="test",
        top_k=100,  # Re-rank top 100 BM25 results
        k_values=[10, 100],  # Compute metrics at k=10 and k=100
        batch_size=32,
    )
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        device="cuda",  # or "cpu"
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:20s}: {metric_value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
