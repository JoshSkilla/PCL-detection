"""
Span-level classifier utilities for PCL detection.

Two-stage pipeline:
1. Train span classifier on Task2 spans (with sampled negatives)
2. Apply to paragraphs via sampling + aggregation for paragraph-level prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# ============================================================================
# Data Leakage Prevention
# ============================================================================

def check_split_leakage(train_df: pd.DataFrame, dev_df: pd.DataFrame, spans_df: pd.DataFrame):
    """
    Check for data leakage: ensure Task2 spans don't leak between train/dev splits.
    
    Returns:
        train_spans: spans from paragraphs in train split only
        dev_spans: spans from paragraphs in dev split only
        stats: dict with leakage diagnostics
    """
    train_par_ids = set(train_df['par_id'].unique())
    dev_par_ids = set(dev_df['par_id'].unique())
    
    # Check for overlap (should be 0)
    overlap = train_par_ids & dev_par_ids
    
    # Filter spans by split
    train_spans = spans_df[spans_df['par_id'].isin(train_par_ids)].copy()
    dev_spans = spans_df[spans_df['par_id'].isin(dev_par_ids)].copy()
    
    stats = {
        'train_par_ids': len(train_par_ids),
        'dev_par_ids': len(dev_par_ids),
        'overlap_par_ids': len(overlap),
        'total_spans': len(spans_df),
        'train_spans': len(train_spans),
        'dev_spans': len(dev_spans),
        'train_unique_pars_with_spans': train_spans['par_id'].nunique(),
        'dev_unique_pars_with_spans': dev_spans['par_id'].nunique(),
    }
    
    return train_spans, dev_spans, stats


# ============================================================================
# Span Dataset Builder
# ============================================================================

def aggregate_multilabel_spans(spans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate spans with same (par_id, start, end) and collapse multilabel.
    Returns one row per unique span with span_bin=1 (positive).
    """
    # Group by exact span location
    grouped = spans_df.groupby(['par_id', 'span_start_norm', 'span_finish_norm']).agg({
        'span_text': 'first',  # Take first text (should all be same)
    }).reset_index()
    
    # All are positive (label_bin=1)
    grouped['span_bin'] = 1
    
    return grouped


def build_span_training_dataset(
    train_df: pd.DataFrame,
    train_spans: pd.DataFrame,
    sampler_fn,
    target_lengths: List[int],
    median_span_len: int,
    negative_ratio: float = 1.0,
    max_spans_per_par: int = 20,
    include_hard_negatives: bool = False,
) -> pd.DataFrame:
    """
    Build balanced span dataset for classifier training.
    
    Positive spans: from Task2 annotations (train split only)
    Negative spans: sampled from non-PCL paragraphs (label_bin=0)
    Optional hard negatives: from PCL paragraphs outside annotated spans
    
    Returns DataFrame with columns: span_text, span_bin, par_id, span_start_char, span_end_char
    """
    # Positive spans (already filtered to train par_ids)
    pos_df = aggregate_multilabel_spans(train_spans)
    pos_df = pos_df.rename(columns={
        'span_start_norm': 'span_start_char',
        'span_finish_norm': 'span_end_char',
    })
    pos_df = pos_df[['span_text', 'span_bin', 'par_id', 'span_start_char', 'span_end_char']]
    
    n_pos = len(pos_df)
    n_neg_needed = int(n_pos * negative_ratio)
    
    # Sample negatives from non-PCL paragraphs
    non_pcl = train_df[train_df['label_bin'] == 0].copy()
    
    neg_spans = []
    for _, row in non_pcl.iterrows():
        if len(neg_spans) >= n_neg_needed:
            break
            
        spans = sampler_fn(
            text=row['text'],
            target_lengths=target_lengths,
            median_span_len=median_span_len,
            max_spans_per_par=max_spans_per_par,
        )
        
        for span in spans:
            neg_spans.append({
                'span_text': span['span_text'],
                'span_bin': 0,
                'par_id': row['par_id'],
                'span_start_char': span['span_start_char'],
                'span_end_char': span['span_end_char'],
            })
            
            if len(neg_spans) >= n_neg_needed:
                break
    
    neg_df = pd.DataFrame(neg_spans)
    
    # Optional: hard negatives from PCL paragraphs (non-overlapping with annotations)
    if include_hard_negatives:
        # TODO: implement if needed later
        pass
    
    # Combine and shuffle
    span_df = pd.concat([pos_df, neg_df], ignore_index=True)
    span_df = span_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return span_df


# ============================================================================
# Span Classifier Model
# ============================================================================

class SpanClassifier(nn.Module):
    """
    Simple span-level binary classifier.
    Takes span text (tokenized), returns binary prediction.
    """
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        
        # Load encoder (with local cache fallback)
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Model load failed ({type(e).__name__}), trying local cache...")
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        
        hidden = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass for span classification.
        
        Returns dict with:
            - loss (if labels provided)
            - logits (B,) binary logits
        """
        # Some models don't use token_type_ids
        try:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        except TypeError:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # CLS token representation
        cls = outputs.last_hidden_state[:, 0]  # (B, H)
        cls = cls.to(dtype=self.classifier.weight.dtype)
        
        cls = self.dropout(cls)
        logits = self.classifier(cls).squeeze(-1)  # (B,)
        
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                labels.float(),
            )
        
        return {
            'loss': loss,
            'logits': logits,
        }


# ============================================================================
# Paragraph Aggregation
# ============================================================================

def aggregate_span_scores(
    span_probs: np.ndarray,
    mode: str = 'max',
    topk: int = 3,
    span_thresh: Optional[float] = None,
) -> float:
    """
    Aggregate span-level probabilities into single paragraph score.
    
    Args:
        span_probs: (K,) array of span probabilities
        mode: 'max', 'topk_mean', 'mean'
        topk: number of top spans to average (for topk_mean)
        span_thresh: optionally filter spans below threshold before aggregation
    
    Returns:
        paragraph_score: float in [0, 1]
    """
    if len(span_probs) == 0:
        return 0.0
    
    # Optional filtering
    if span_thresh is not None:
        span_probs = span_probs[span_probs >= span_thresh]
        if len(span_probs) == 0:
            return 0.0
    
    if mode == 'max':
        return float(np.max(span_probs))
    
    elif mode == 'topk_mean':
        k = min(topk, len(span_probs))
        top_indices = np.argpartition(span_probs, -k)[-k:]
        return float(np.mean(span_probs[top_indices]))
    
    elif mode == 'mean':
        return float(np.mean(span_probs))
    
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def evaluate_paragraph_predictions(
    model: nn.Module,
    paragraph_df: pd.DataFrame,
    sampler_fn,
    target_lengths: List[int],
    median_span_len: int,
    tokenizer,
    max_len: int,
    device: str,
    use_amp: bool,
    amp_dtype,
    num_spans_per_par: int = 20,
    agg_mode: str = 'max',
    topk: int = 3,
    span_thresh: Optional[float] = None,
    para_thresh: float = 0.5,
) -> Dict:
    """
    Evaluate model on paragraphs using span sampling + aggregation.
    
    For each paragraph:
    1. Sample K spans using sampler
    2. Score each span with model
    3. Aggregate span scores to paragraph prediction
    4. Compute paragraph-level metrics
    
    Returns dict with metrics: f1, precision, recall, accuracy
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _, row in paragraph_df.iterrows():
            # Sample spans from paragraph
            spans = sampler_fn(
                text=row['text'],
                target_lengths=target_lengths,
                median_span_len=median_span_len,
                max_spans_per_par=num_spans_per_par,
            )
            
            if len(spans) == 0:
                # No spans sampled -> predict negative
                para_score = 0.0
            else:
                # Tokenize all spans
                span_texts = [s['span_text'] for s in spans]
                encodings = tokenizer(
                    span_texts,
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                
                # Move to device
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                token_type_ids = encodings.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                # Score spans
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                
                # Get probabilities
                logits = outputs['logits'].detach().cpu().numpy()
                span_probs = 1.0 / (1.0 + np.exp(-logits))
                
                # Aggregate to paragraph score
                para_score = aggregate_span_scores(
                    span_probs,
                    mode=agg_mode,
                    topk=topk,
                    span_thresh=span_thresh,
                )
            
            # Apply paragraph threshold
            para_pred = 1 if para_score >= para_thresh else 0
            
            all_preds.append(para_pred)
            all_labels.append(int(row['label_bin']))
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'f1': f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'precision': precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'recall': recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'accuracy': accuracy_score(all_labels, all_preds),
        'pos_acc': accuracy_score(all_labels[all_labels == 1], all_preds[all_labels == 1]) if np.any(all_labels == 1) else 0.0,
        'neg_acc': accuracy_score(all_labels[all_labels == 0], all_preds[all_labels == 0]) if np.any(all_labels == 0) else 0.0,
    }
    
    return metrics


def compute_span_metrics(
    model: nn.Module,
    span_df: pd.DataFrame,
    tokenizer,
    max_len: int,
    device: str,
    use_amp: bool,
    amp_dtype,
    batch_size: int = 64,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute span-level metrics on a span dataset.
    Used for diagnostic/monitoring purposes (not the optimization target).
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    
    # Tokenize all spans
    encodings = tokenizer(
        span_df['span_text'].tolist(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    labels = torch.tensor(span_df['span_bin'].values, dtype=torch.float32)
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        encodings.get('token_type_ids', torch.zeros_like(encodings['input_ids'])),
        labels,
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids, batch_labels = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None
            
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            
            all_logits.append(outputs['logits'].detach().cpu())
            all_labels.append(batch_labels)
    
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        'f1': f1_score(labels, preds, pos_label=1, zero_division=0),
        'precision': precision_score(labels, preds, pos_label=1, zero_division=0),
        'recall': recall_score(labels, preds, pos_label=1, zero_division=0),
        'accuracy': accuracy_score(labels, preds),
        'pos_acc': accuracy_score(labels[labels == 1], preds[labels == 1]) if np.any(labels == 1) else 0.0,
        'neg_acc': accuracy_score(labels[labels == 0], preds[labels == 0]) if np.any(labels == 0) else 0.0,
    }
    
    return metrics
