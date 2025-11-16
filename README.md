# rank-validator

A faster in-training validation pipeline for HuggingFace Trainer that uses ranking metrics aligned with final IR evaluation.

## Features

- 🚀 **Seamless Integration**: Drop-in callback for HuggingFace Trainer
- 📊 **Ranking Metrics**: Compute nDCG, MRR, MAP, and Recall during training
- 🔄 **BM25 Re-ranking**: Uses pre-retrieved BM25 top-k results for efficient evaluation
- 🤗 **HuggingFace Datasets**: Automatically loads IR test sets from HuggingFace Hub
- ⚡ **Fast Evaluation**: Batch processing with GPU acceleration

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Using with HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments
from rank_validator import RankValidationCallback

# Create the validation callback
callback = RankValidationCallback(
    dataset_name="your-username/ir-test-dataset",
    split="dev",
    top_k=100,
    k_values=[10, 100],
    batch_size=32,
)

# Add to your trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[callback],
)

trainer.train()
```

### Standalone Evaluation

```python
from rank_validator import RankEvaluator

evaluator = RankEvaluator(
    dataset_name="your-username/ir-test-dataset",
    split="test",
    top_k=100,
    k_values=[10, 100],
)

metrics = evaluator.evaluate(model=model, tokenizer=tokenizer, device="cuda")
print(metrics)
```

## Dataset Format

Your HuggingFace dataset should have the following configurations:

### 1. Queries (`queries`)
- Columns: `qid` (str), `text` (str)

### 2. Corpus (`corpus`)
- Columns: `docid` (str), `text` (str)

### 3. Query Relevance Judgments (`qrels`)
- Columns: `qid` (str), `docid` (str), `relevance` (int)

### 4. BM25 Results (`bm25_results`)
- Columns: `qid` (str), and either:
  - `docids` (list[str]), `scores` (list[float]), OR
  - `results` (dict mapping docid to score)

## Metrics Computed

The validation pipeline uses the [ir_measures](https://github.com/terrierteam/ir_measures) library to compute ranking metrics:

- **nDCG@k**: Normalized Discounted Cumulative Gain
- **Recall@k**: Recall at cutoff k
- **MRR**: Mean Reciprocal Rank
- **MAP**: Mean Average Precision

All metrics are computed using the provided relevance judgments and model re-rankings of the BM25 top-k results. The `ir_measures` library ensures accurate and standardized metric computation aligned with TREC and other IR evaluation benchmarks.

## Examples

See the `examples/` directory for complete examples:

- `train_with_validation.py`: Training with in-training validation
- `standalone_evaluation.py`: Standalone evaluation without training

## Configuration Options

### RankValidationCallback

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | HuggingFace dataset name |
| `split` | str | `"dev"` | Dataset split to use |
| `top_k` | int | `100` | Number of BM25 results to re-rank |
| `k_values` | List[int] | `[10, 100]` | K values for metrics@k |
| `cache_dir` | str | `None` | Directory to cache datasets |
| `batch_size` | int | `32` | Batch size for inference |
| `eval_steps` | int | `None` | Evaluate every N steps |
| `prefix` | str | `"eval"` | Prefix for logged metrics |

### RankEvaluator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | HuggingFace dataset name |
| `split` | str | `"test"` | Dataset split to use |
| `top_k` | int | `100` | Number of BM25 results to re-rank |
| `k_values` | List[int] | `[10, 100]` | K values for metrics@k |
| `cache_dir` | str | `None` | Directory to cache datasets |
| `batch_size` | int | `32` | Batch size for inference |

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- Datasets >= 2.0.0
- NumPy >= 1.19.0
- tqdm >= 4.62.0
- ir-measures >= 0.3.0

## License

MIT License

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{rank_validator,
  title = {rank-validator: A faster in-training validation pipeline},
  author = {Dylan Joo},
  year = {2025},
  url = {https://github.com/DylanJoo/rank-validator}
}
```
