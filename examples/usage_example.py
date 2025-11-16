"""
Example usage of QrelDataset for in-training evaluation.

This example shows how to use the QrelDataset class to load evaluation data
alongside training data, enabling faster validation during training.
"""

from dataclasses import dataclass, field
from typing import Optional
from rank_validator.dataset_dev import QrelDataset


@dataclass
class DataArguments:
    """
    Arguments for dataset configuration.
    
    This class contains the necessary arguments to configure both training
    and evaluation datasets. The eval_* prefixed arguments are specifically
    for the evaluation dataset used by QrelDataset.
    """
    # Training dataset arguments
    dataset_name: str = field(
        default='json', 
        metadata={"help": "huggingface dataset name for training"}
    )
    dataset_split: str = field(
        default='train', 
        metadata={"help": "dataset split for training"}
    )
    corpus_name: Optional[str] = field(
        default=None, 
        metadata={"help": "huggingface dataset name for training corpus"}
    )
    train_group_size: int = field(
        default=8, 
        metadata={"help": "number of passages per query for training"}
    )
    
    # Evaluation dataset arguments (for QrelDataset)
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "huggingface dataset name for evaluation qrels"}
    )
    eval_dataset_split: str = field(
        default='validation',
        metadata={"help": "dataset split for evaluation"}
    )
    eval_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "dataset config for evaluation (e.g., specific qrel set)"}
    )
    eval_corpus_name: Optional[str] = field(
        default=None,
        metadata={"help": "huggingface dataset name for evaluation corpus"}
    )
    eval_corpus_config: Optional[str] = field(
        default=None,
        metadata={"help": "corpus config for evaluation"}
    )
    eval_group_size: int = field(
        default=8,
        metadata={"help": "number of passages per query for evaluation"}
    )
    
    # Text processing arguments
    query_max_len: int = field(
        default=32,
        metadata={"help": "maximum query length"}
    )
    passage_max_len: int = field(
        default=128,
        metadata={"help": "maximum passage length"}
    )
    query_prefix: str = field(
        default='',
        metadata={"help": "prefix for query text"}
    )
    passage_prefix: str = field(
        default='',
        metadata={"help": "prefix for passage text"}
    )
    
    # Dataset loading arguments
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "cache directory for datasets"}
    )
    corpus_split: str = field(
        default='train',
        metadata={"help": "corpus split to use"}
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "number of processes for dataset loading"}
    )
    
    # Multimodal encoding flags
    encode_text: bool = field(
        default=True,
        metadata={"help": "whether to encode text"}
    )
    encode_image: bool = field(
        default=True,
        metadata={"help": "whether to encode images"}
    )
    encode_video: bool = field(
        default=True,
        metadata={"help": "whether to encode videos"}
    )
    encode_audio: bool = field(
        default=True,
        metadata={"help": "whether to encode audio"}
    )


def example_basic_usage():
    """
    Basic example of using QrelDataset.
    """
    print("=" * 80)
    print("Example 1: Basic QrelDataset Usage")
    print("=" * 80)
    
    # Configure data arguments
    data_args = DataArguments(
        eval_dataset_name='DylanJHJ/Qrels',
        eval_dataset_split='msmarco_passage.trec_dl_2019',
        eval_corpus_name='Tevatron/msmarco-passage-corpus-new',
        eval_group_size=8,
        query_prefix='query: ',
        passage_prefix='passage: ',
    )
    
    # Create the evaluation dataset
    print(f"Loading evaluation dataset: {data_args.eval_dataset_name}")
    print(f"  Split: {data_args.eval_dataset_split}")
    print(f"  Corpus: {data_args.eval_corpus_name}")
    print(f"  Group size: {data_args.eval_group_size}")
    
    try:
        eval_dataset = QrelDataset(data_args)
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Total evaluation queries: {len(eval_dataset)}")
        
        # Get a sample
        if len(eval_dataset) > 0:
            query, passages = eval_dataset[0]
            query_text, query_img, query_vid, query_aud = query
            print(f"\nSample query: {query_text[:100]}...")
            print(f"Number of passages: {len(passages)}")
            if len(passages) > 0:
                passage_text, p_img, p_vid, p_aud = passages[0]
                print(f"First passage: {passage_text[:100]}...")
    
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        print("  This is expected if you don't have the dataset downloaded.")


def example_with_training():
    """
    Example showing how to use QrelDataset alongside training data.
    """
    print("\n" + "=" * 80)
    print("Example 2: Using QrelDataset with Training Data")
    print("=" * 80)
    
    # Configure both training and evaluation
    data_args = DataArguments(
        # Training data
        dataset_name='your-training-dataset',
        dataset_split='train',
        corpus_name='your-training-corpus',
        train_group_size=8,
        
        # Evaluation data (for in-training validation)
        eval_dataset_name='your-eval-qrels',
        eval_dataset_split='validation',
        eval_corpus_name='your-eval-corpus',
        eval_group_size=8,
        
        # Text processing
        query_max_len=32,
        passage_max_len=512,
        query_prefix='',
        passage_prefix='',
    )
    
    print("Configuration for in-training validation:")
    print(f"  Training dataset: {data_args.dataset_name}")
    print(f"  Evaluation dataset: {data_args.eval_dataset_name}")
    print(f"  This allows validation during training without stopping!")


def example_pretokenization():
    """
    Example showing how to use QrelDataset for pre-tokenization,
    similar to the SCOPE repository usage.
    """
    print("\n" + "=" * 80)
    print("Example 3: Pre-tokenization for Faster Training")
    print("=" * 80)
    
    print("""
    You can pre-tokenize both training and evaluation datasets:
    
    ```python
    from transformers import AutoTokenizer
    from datasets import Dataset, DatasetDict
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('your-model')
    
    # Create QrelDataset
    data_args = DataArguments(
        eval_dataset_name='your-qrels',
        eval_corpus_name='your-corpus',
        query_max_len=32,
        passage_max_len=512,
    )
    eval_dataset = QrelDataset(data_args)
    
    # Convert to HuggingFace format and tokenize
    def convert_and_tokenize(i):
        query, passages = eval_dataset[i]
        query_text = query[0]
        passage_texts = [p[0] for p in passages]
        
        return {
            'query_tokenized': tokenizer(query_text, ...)['input_ids'],
            'passage_tokenized': tokenizer(passage_texts, ...)['input_ids'],
        }
    
    # Apply to all items
    tokenized_dataset = Dataset.from_list([
        convert_and_tokenize(i) for i in range(len(eval_dataset))
    ])
    
    # Save for later use
    tokenized_dataset.save_to_disk('path/to/save')
    ```
    
    This pre-tokenization approach:
    - Speeds up training by avoiding repeated tokenization
    - Enables efficient data loading
    - Works well with both training and evaluation datasets
    """)


if __name__ == "__main__":
    print("Rank-Validator: In-Training Validation Examples")
    print("=" * 80)
    print()
    print("This script demonstrates how to use QrelDataset for faster")
    print("in-training validation by loading evaluation data alongside")
    print("training data.")
    print()
    
    # Run examples
    example_basic_usage()
    example_with_training()
    example_pretokenization()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
