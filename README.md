# rank-validator

A faster in-training validation pipeline for information retrieval models.

## Overview

`rank-validator` provides efficient in-training validation for neural ranking models by implementing the evaluation dataset loading logic from Tevatron. This allows you to validate your models during training without interrupting the training loop, leading to faster experimentation and better model development.

## Features

- **QrelDataset**: Load evaluation datasets (qrels format) alongside training data
- **Efficient Data Loading**: Uses HuggingFace datasets for fast, cached data loading
- **Multimodal Support**: Handle text, images, video, and audio in both queries and documents
- **Pre-tokenization**: Support for pre-tokenizing datasets to speed up training
- **Compatible with Tevatron**: Works seamlessly with the Tevatron framework

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install torch datasets transformers
```

## Quick Start

### Basic Usage

```python
from dataclasses import dataclass, field
from typing import Optional
from rank_validator.dataset_dev import QrelDataset


@dataclass
class DataArguments:
    eval_dataset_name: str = 'DylanJHJ/Qrels'
    eval_dataset_split: str = 'msmarco_passage.trec_dl_2019'
    eval_corpus_name: str = 'Tevatron/msmarco-passage-corpus-new'
    eval_group_size: int = 8
    query_prefix: str = ''
    passage_prefix: str = ''
    # ... other arguments


# Create evaluation dataset
data_args = DataArguments()
eval_dataset = QrelDataset(data_args)

# Get query and passages
query, passages = eval_dataset[0]
query_text, query_img, query_vid, query_aud = query
```

### In-Training Validation

Use `QrelDataset` alongside your training dataset to enable validation during training:

```python
from rank_validator.dataset_dev import QrelDataset

# Configure evaluation data
data_args = DataArguments(
    # Training data
    dataset_name='your-training-dataset',
    corpus_name='your-training-corpus',
    
    # Evaluation data
    eval_dataset_name='your-eval-qrels',
    eval_corpus_name='your-eval-corpus',
    eval_group_size=8,
)

# Create datasets
train_dataset = TrainDataset(data_args)  # Your training dataset
eval_dataset = QrelDataset(data_args)    # Evaluation dataset

# Use in your training loop
# The eval_dataset can be used at any point during training
# without loading data from scratch each time
```

### Pre-tokenization

Pre-tokenize your datasets for even faster training:

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained('your-model')
eval_dataset = QrelDataset(data_args)

def convert_and_tokenize(i):
    query, passages = eval_dataset[i]
    query_text = query[0]
    passage_texts = [p[0] for p in passages]
    
    return {
        'query_tokenized': tokenizer(query_text, max_length=32)['input_ids'],
        'passage_tokenized': tokenizer(passage_texts, max_length=512)['input_ids'],
    }

tokenized_dataset = Dataset.from_list([
    convert_and_tokenize(i) for i in range(len(eval_dataset))
])

tokenized_dataset.save_to_disk('path/to/save')
```

## DataArguments

The `QrelDataset` requires a `DataArguments` object with the following evaluation-specific fields:

### Required Arguments

- `eval_dataset_name`: HuggingFace dataset name for evaluation qrels
- `eval_corpus_name`: HuggingFace dataset name for evaluation corpus

### Optional Arguments

- `eval_dataset_split`: Dataset split for evaluation (default: 'validation')
- `eval_dataset_config`: Dataset config for specific qrel sets
- `eval_corpus_config`: Corpus config for evaluation
- `eval_group_size`: Number of passages per query (default: 8)
- `query_prefix`: Prefix for query text (e.g., "query: ")
- `passage_prefix`: Prefix for passage text (e.g., "passage: ")
- `dataset_cache_dir`: Cache directory for datasets
- `corpus_split`: Corpus split to use (default: 'train')
- `num_proc`: Number of processes for dataset loading (default: 1)
- `encode_text`, `encode_image`, `encode_video`, `encode_audio`: Flags for multimodal encoding

## Examples

See the `examples/` directory for complete usage examples:

- `examples/usage_example.py`: Comprehensive examples showing different use cases

Run the examples:

```bash
python examples/usage_example.py
```

## Background

This implementation is adapted from the evaluation dataset logic used in the SCOPE repository, which extends the Tevatron framework. The key innovation is the `QrelDataset` class that:

1. Loads qrels (query relevance judgments) from HuggingFace datasets
2. Maps docids to corpus entries efficiently using a hash map
3. Supports epoch-based sampling for consistent validation
4. Works with both text and multimodal content

## License

MIT License

## Citation

If you use this code, please cite the original Tevatron framework:

```bibtex
@software{tevatron,
  author = {Gao, Luyu and others},
  title = {Tevatron: An efficient and flexible toolkit for dense retrieval research},
  url = {https://github.com/texttron/tevatron},
  year = {2021}
}
```

## Acknowledgments

This implementation is based on the evaluation dataset logic from the user's forked Tevatron repository, as referenced in the SCOPE project.
