"""
Integration example mimicking the SCOPE repository usage.

This example demonstrates how to use QrelDataset in a similar way to how it's
used in the SCOPE repository's pretokenization script.
"""

from dataclasses import dataclass, field
from typing import Optional
import sys

# Add src to path for local development
sys.path.insert(0, 'src')

from rank_validator.dataset_dev import QrelDataset


@dataclass
class DataArguments:
    """
    Complete DataArguments matching the SCOPE usage pattern.
    """
    # Training dataset
    dataset_name: str = field(
        default='DylanJHJ/crux-researchy',
        metadata={"help": "Training dataset name"}
    )
    dataset_split: str = field(
        default='flatten',
        metadata={"help": "Training dataset split"}
    )
    corpus_name: str = field(
        default='DylanJHJ/crux-researchy-corpus',
        metadata={"help": "Training corpus name"}
    )
    train_group_size: int = field(
        default=8,
        metadata={"help": "Number of passages per query in training"}
    )
    
    # Evaluation dataset (for QrelDataset)
    eval_dataset_name: str = field(
        default='DylanJHJ/Qrels',
        metadata={"help": "Evaluation qrels dataset name"}
    )
    eval_dataset_split: str = field(
        default='msmarco_passage.trec_dl_2019',
        metadata={"help": "Evaluation dataset split"}
    )
    eval_corpus_name: str = field(
        default='Tevatron/msmarco-passage-corpus-new',
        metadata={"help": "Evaluation corpus name"}
    )
    eval_group_size: int = field(
        default=8,
        metadata={"help": "Number of passages per query in evaluation"}
    )
    
    # Text configuration
    exclude_title: bool = field(
        default=True,
        metadata={"help": "Whether to exclude title from passages"}
    )
    query_max_len: int = field(
        default=32,
        metadata={"help": "Maximum query length"}
    )
    passage_max_len: int = field(
        default=512,
        metadata={"help": "Maximum passage length"}
    )
    query_prefix: str = field(
        default='',
        metadata={"help": "Query prefix/instruction"}
    )
    passage_prefix: str = field(
        default='',
        metadata={"help": "Passage prefix/instruction"}
    )
    append_eos_token: bool = field(
        default=False,
        metadata={"help": "Append EOS token"}
    )
    
    # Dataset loading
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory"}
    )
    corpus_split: str = field(
        default='train',
        metadata={"help": "Corpus split"}
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "Number of processes"}
    )
    
    # Encoding flags
    encode_text: bool = field(default=True)
    encode_image: bool = field(default=True)
    encode_video: bool = field(default=True)
    encode_audio: bool = field(default=True)


def main():
    """
    Main function demonstrating the SCOPE-style usage.
    """
    print("=" * 80)
    print("SCOPE-style Integration Example")
    print("=" * 80)
    print()
    
    # Configure data arguments (matching SCOPE usage)
    data_args = DataArguments(
        exclude_title=True,
        dataset_name='DylanJHJ/crux-researchy',
        dataset_split='flatten',
        corpus_name='DylanJHJ/crux-researchy-corpus',
        train_group_size=8,
        query_max_len=32,
        passage_max_len=512,
        eval_dataset_name='DylanJHJ/Qrels',
        eval_dataset_split='msmarco_passage.trec_dl_2019',
        eval_corpus_name='Tevatron/msmarco-passage-corpus-new',
        eval_group_size=8,
    )
    
    print("Configuration:")
    print(f"  Training dataset: {data_args.dataset_name}")
    print(f"  Training corpus: {data_args.corpus_name}")
    print(f"  Evaluation dataset: {data_args.eval_dataset_name}")
    print(f"  Evaluation corpus: {data_args.eval_corpus_name}")
    print(f"  Query max length: {data_args.query_max_len}")
    print(f"  Passage max length: {data_args.passage_max_len}")
    print()
    
    # Create QrelDataset for evaluation
    print("Loading evaluation dataset...")
    try:
        # Note: This will attempt to download the dataset from HuggingFace
        # In SCOPE, this is combined with TrainDataset for both splits
        eval_dataset = QrelDataset(
            data_args,
            corpus_name=data_args.eval_corpus_name
        )
        
        print(f"✓ Evaluation dataset loaded successfully!")
        print(f"  Total queries: {len(eval_dataset)}")
        print()
        
        # Demonstrate getting a sample
        if len(eval_dataset) > 0:
            print("Sample from evaluation dataset:")
            query, passages = eval_dataset[0]
            query_text, query_img, query_vid, query_aud = query
            
            print(f"  Query: {query_text[:80]}...")
            print(f"  Number of passages: {len(passages)}")
            
            if len(passages) > 0:
                passage_text, p_img, p_vid, p_aud = passages[0]
                print(f"  First passage: {passage_text[:80]}...")
        
        print()
        print("=" * 80)
        print("Usage Notes:")
        print("=" * 80)
        print("""
This QrelDataset can be used alongside TrainDataset in your training loop:

```python
from tevatron.retriever.dataset import TrainDataset
from rank_validator.dataset_dev import QrelDataset

# Load both training and evaluation datasets
train_dataset = TrainDataset(data_args)
eval_dataset = QrelDataset(data_args, corpus_name=data_args.eval_corpus_name)

# Use in training
for split in ['train', 'eval']:
    if split == 'train':
        dataset = train_dataset
    else:
        dataset = eval_dataset
    
    # Process dataset...
    for i in range(len(dataset)):
        query, passages = dataset[i]
        # Your training/evaluation logic here
```

This enables in-training validation without interrupting the training loop!
        """)
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print()
        print("This is expected if:")
        print("  - The datasets library is not installed")
        print("  - You don't have network access")
        print("  - The specified datasets don't exist on HuggingFace")
        print()
        print("To use with your own data, simply modify the dataset names")
        print("in the DataArguments configuration above.")


if __name__ == "__main__":
    main()
