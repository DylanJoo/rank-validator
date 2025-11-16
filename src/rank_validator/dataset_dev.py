"""
Evaluation dataset module for in-training validation.

This module implements QrelDataset which loads evaluation data (qrels format)
to enable in-training validation. This is adapted from the Tevatron framework
to support evaluation datasets alongside training data.
"""

import random
from typing import Optional
from datasets import load_dataset
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class QrelDataset(Dataset):
    """
    Dataset for evaluation which handles qrels (query relevance judgments).
    
    This dataset loads evaluation data in qrels format and creates query-passage
    pairs for in-training validation. It's designed to work alongside TrainDataset
    to enable faster validation during training.
    
    Args:
        data_args: DataArguments containing dataset configuration
        corpus_name: Optional override for corpus name (defaults to eval_corpus_name from data_args)
        trainer: Optional trainer instance for epoch-based sampling
    """

    def __init__(self,
                 data_args,
                 corpus_name: Optional[str] = None,
                 trainer=None):
        self.data_args = data_args
        self.trainer = trainer
        
        # Get evaluation dataset parameters
        eval_dataset_name = getattr(data_args, 'eval_dataset_name', None)
        eval_dataset_split = getattr(data_args, 'eval_dataset_split', 'validation')
        eval_corpus_name = corpus_name if corpus_name is not None else getattr(data_args, 'eval_corpus_name', None)
        eval_group_size = getattr(data_args, 'eval_group_size', getattr(data_args, 'train_group_size', 8))
        
        if eval_dataset_name is None:
            raise ValueError("eval_dataset_name must be provided in data_args")
        
        # Load evaluation qrels dataset
        self.eval_data = load_dataset(
            eval_dataset_name,
            getattr(data_args, 'eval_dataset_config', None),
            split=eval_dataset_split,
            cache_dir=getattr(data_args, 'dataset_cache_dir', None),
            num_proc=getattr(data_args, 'num_proc', 1),
        )
        
        # Load evaluation corpus
        if eval_corpus_name is None:
            raise ValueError("eval_corpus_name must be provided")
        
        self.corpus = load_dataset(
            eval_corpus_name,
            getattr(data_args, 'eval_corpus_config', None),
            split=getattr(data_args, 'corpus_split', 'train'),
            cache_dir=getattr(data_args, 'dataset_cache_dir', None),
            num_proc=getattr(data_args, 'num_proc', 1),
        )
        
        # Create docid to index mapping for fast lookup
        self.docid_to_index = {}
        if self.corpus is not None:
            corpus_ids = self.corpus.select_columns(['docid'])
            docids = corpus_ids['docid']
            self.docid_to_index = {docid: index for index, docid in enumerate(docids)}
        
        self.eval_group_size = eval_group_size

    def set_trainer(self, trainer):
        """Sets the trainer for the dataset."""
        self.trainer = trainer

    def __len__(self):
        return len(self.eval_data)

    def _get_info_from_docid(self, docid, prefix=""):
        """
        Retrieves document information from the corpus given a docid.
        
        Args:
            docid: Document ID to retrieve
            prefix: Prefix to add to the text (e.g., "passage: ")
        
        Returns:
            tuple: (formatted_text, image, video, audio)
        """
        document_info = self.corpus[self.docid_to_index[docid]]
        assert document_info['docid'] == docid
        
        # Get multimodal content
        image = document_info.get('image', None)
        video = document_info.get('video', None)
        audio = document_info.get('audio', None)
        text = document_info.get('text', '')
        
        # Apply encoding flags if available
        encode_text = getattr(self.data_args, 'encode_text', True)
        encode_image = getattr(self.data_args, 'encode_image', True)
        encode_video = getattr(self.data_args, 'encode_video', True)
        encode_audio = getattr(self.data_args, 'encode_audio', True)
        
        if not encode_text:
            text = None
        if not encode_image:
            image = None
        if not encode_video:
            video = None
        if not encode_audio:
            audio = None
        
        text = '' if text is None else text
        return prefix + text, image, video, audio

    def __getitem__(self, item):
        """
        Get a query and its associated passages for evaluation.
        
        Args:
            item: Index of the item to retrieve
        
        Returns:
            tuple: (formatted_query, formatted_documents)
                formatted_query: (query_text, query_image, query_video, query_audio)
                formatted_documents: List of (doc_text, doc_image, doc_video, doc_audio)
        """
        group = self.eval_data[item]
        
        # Get query information
        query_id = group.get('query_id', None)
        query_text = group.get('query_text', group.get('query', '')) or ''
        query_image = group.get('query_image', None)
        query_video = group.get('query_video', None)
        query_audio = group.get('query_audio', None)
        
        query_prefix = getattr(self.data_args, 'query_prefix', '')
        passage_prefix = getattr(self.data_args, 'passage_prefix', '')
        
        formatted_query = (
            query_prefix + query_text,
            query_image,
            query_video,
            query_audio
        )
        
        # Get document IDs from qrels
        # Support both positive/negative format and qrel format
        if 'positive_document_ids' in group:
            positive_document_ids = group['positive_document_ids']
            negative_document_ids = group.get('negative_document_ids', [])
        elif 'docids' in group:
            # Alternative format: all docids with relevance scores
            positive_document_ids = group['docids']
            negative_document_ids = []
        else:
            raise ValueError("Evaluation data must contain either 'positive_document_ids' or 'docids'")
        
        formatted_documents = []
        
        # Sample positive documents
        if len(positive_document_ids) > 0:
            # Use epoch-based sampling if trainer is available
            if self.trainer is not None:
                epoch = int(self.trainer.state.epoch)
                _hashed_seed = hash(item + self.trainer.args.seed)
                selected_positive_docid = positive_document_ids[
                    (_hashed_seed + epoch) % len(positive_document_ids)
                ]
            else:
                # Random selection if no trainer
                selected_positive_docid = random.choice(positive_document_ids)
            
            formatted_documents.append(
                self._get_info_from_docid(selected_positive_docid, passage_prefix)
            )
        
        # Sample negative documents to fill eval_group_size
        negative_size = self.eval_group_size - len(formatted_documents)
        if negative_size > 0 and len(negative_document_ids) > 0:
            if len(negative_document_ids) < negative_size:
                selected_negative_docids = random.choices(
                    negative_document_ids, 
                    k=negative_size
                )
            else:
                if self.trainer is not None:
                    epoch = int(self.trainer.state.epoch)
                    _hashed_seed = hash(item + self.trainer.args.seed)
                    offset = epoch * negative_size % len(negative_document_ids)
                    selected_negative_docids = list(negative_document_ids)
                    random.Random(_hashed_seed).shuffle(selected_negative_docids)
                    selected_negative_docids = selected_negative_docids * 2
                    selected_negative_docids = selected_negative_docids[offset: offset + negative_size]
                else:
                    selected_negative_docids = random.sample(
                        negative_document_ids,
                        min(negative_size, len(negative_document_ids))
                    )
            
            for neg_docid in selected_negative_docids:
                formatted_documents.append(
                    self._get_info_from_docid(neg_docid, passage_prefix)
                )
        
        return formatted_query, formatted_documents
