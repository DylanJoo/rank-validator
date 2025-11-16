"""
HuggingFace Trainer callback for in-training validation with ranking metrics.
"""

from typing import Dict, List, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from .evaluator import RankEvaluator


class RankValidationCallback(TrainerCallback):
    """
    Callback for HuggingFace Trainer to perform in-training validation with ranking metrics.
    
    This callback:
    1. Loads IR test datasets from HuggingFace
    2. Uses BM25 top-k results for re-ranking
    3. Evaluates the model with ranking metrics during training
    4. Logs metrics to the trainer's logging system
    
    Example:
        ```python
        from transformers import Trainer
        from rank_validator import RankValidationCallback
        
        callback = RankValidationCallback(
            dataset_name="your-username/ir-dataset",
            split="dev",
            top_k=100,
            k_values=[10, 100],
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[callback],
        )
        ```
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "dev",
        top_k: int = 100,
        k_values: List[int] = [10, 100],
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        eval_steps: Optional[int] = None,
        prefix: str = "eval",
    ):
        """
        Initialize the rank validation callback.
        
        Args:
            dataset_name: Name of the HuggingFace IR dataset
            split: Dataset split to use for validation
            top_k: Number of top BM25 results to re-rank
            k_values: List of k values for metrics@k (e.g., [10, 100])
            cache_dir: Directory to cache datasets
            batch_size: Batch size for model inference
            eval_steps: Evaluate every N steps (None = use trainer's eval_steps)
            prefix: Prefix for logging metrics (e.g., "eval", "dev")
        """
        self.dataset_name = dataset_name
        self.split = split
        self.top_k = top_k
        self.k_values = k_values
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.prefix = prefix
        
        self.evaluator = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of the evaluator."""
        if not self._initialized:
            print(f"\n{'='*80}")
            print(f"Initializing RankValidationCallback")
            print(f"Dataset: {self.dataset_name} ({self.split})")
            print(f"Top-k: {self.top_k}, K-values: {self.k_values}")
            print(f"{'='*80}\n")
            
            self.evaluator = RankEvaluator(
                dataset_name=self.dataset_name,
                split=self.split,
                top_k=self.top_k,
                k_values=self.k_values,
                cache_dir=self.cache_dir,
                batch_size=self.batch_size,
            )
            self._initialized = True
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        tokenizer=None,
        **kwargs,
    ):
        """
        Called after evaluation phase during training.
        
        This method performs ranking evaluation and logs the metrics.
        """
        # Initialize evaluator on first call
        if not self._initialized:
            self._initialize()
        
        # Check if we should evaluate based on eval_steps
        if self.eval_steps is not None:
            if state.global_step % self.eval_steps != 0:
                return
        
        print(f"\n{'='*80}")
        print(f"Running ranking validation at step {state.global_step}")
        print(f"{'='*80}\n")
        
        # Evaluate with ranking metrics
        device = args.device if hasattr(args, 'device') else 'cuda'
        metrics = self.evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        
        # Add prefix to metrics
        prefixed_metrics = {
            f"{self.prefix}_{k}": v for k, v in metrics.items()
        }
        
        # Log metrics
        if state.is_world_process_zero:
            print(f"\nRanking Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            print()
            
            # Add to state logs
            if state.log_history:
                state.log_history[-1].update(prefixed_metrics)
            
        return control
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the beginning of training.
        
        Pre-load the dataset to avoid delays during first evaluation.
        """
        self._initialize()
        return control
