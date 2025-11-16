"""
Example script showing how to use RankValidationCallback with HuggingFace Trainer.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from rank_validator import RankValidationCallback


def main():
    # Load your model and tokenizer
    model_name = "bert-base-uncased"  # Replace with your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # For ranking/regression
    )
    
    # Load your training dataset
    # This is just an example - replace with your actual training data
    train_dataset = load_dataset("your-training-dataset", split="train")
    
    # Create the rank validation callback
    callback = RankValidationCallback(
        dataset_name="your-username/ir-test-dataset",  # Replace with your IR dataset
        split="dev",  # or "test"
        top_k=100,  # Use top 100 BM25 results
        k_values=[10, 100],  # Compute metrics@10 and metrics@100
        batch_size=32,
        prefix="eval",  # Prefix for logged metrics
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps
        logging_steps=100,
        save_steps=1000,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )
    
    # Create trainer with the callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[callback],
    )
    
    # Train the model
    trainer.train()
    
    print("\nTraining completed!")
    print("Ranking metrics were computed during training and logged.")


if __name__ == "__main__":
    main()
