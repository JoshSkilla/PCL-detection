"""
Deterministic Span Sampler for PCL Detection

This module provides functions for deterministically sampling text spans from paragraphs
for the span-first training strategy in Patronizing and Condescending Language (PCL) detection.

Main functions:
- sample_spans_for_paragraph: Core sampling function
- build_span_training_dataset: Create training dataset with positive/negative spans
- build_eval_span_candidates: Generate evaluation span candidates
"""

import spacy
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Load spaCy for sentence segmentation (module-level singleton)
_nlp = None

def get_spacy_model():
    """Get or load the spaCy model (singleton pattern)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
    return _nlp


# Simple word tokenizer for consistency
def word_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer matching EDA approach."""
    return str(text).split()


@dataclass
class TextUnit:
    """Represents a segmented unit of text (sentence or clause)."""
    start_char: int
    end_char: int
    text: str
    token_len: int


@dataclass
class SpanAnnotation:
    """Represents a sampled span."""
    start_char: int
    end_char: int
    text: str
    token_len: int
    anchor_idx: int
    units_covered: List[int]


def segment_paragraph_into_units(text: str) -> List[TextUnit]:
    """
    Segment paragraph using spaCy sentences + optional clause splitting.
    
    Splits on safe phrase boundaries: ", ", "; ", " — " (em dash with spaces).
    Does NOT split on periods (spaCy handles abbreviations) or hyphens in words.
    
    Args:
        text: Paragraph text to segment
        
    Returns:
        List of TextUnit objects with char offsets and token counts.
    """
    nlp = get_spacy_model()
    units = []
    
    # First: spaCy sentence segmentation
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # Safe delimiter patterns for clause splitting (with surrounding whitespace)
    clause_delimiters = [
        (r'\s+,\s+', ', '),      # comma with spaces
        (r'\s+;\s+', '; '),      # semicolon with spaces  
        (r'\s+—\s+', ' — '),     # em dash with spaces
        (r'\s+–\s+', ' – '),     # en dash with spaces
    ]
    
    for sent in sentences:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_text = text[sent_start:sent_end]
        
        # Try to split into clauses
        clause_parts = [sent_text]
        for pattern, delimiter in clause_delimiters:
            new_parts = []
            for part in clause_parts:
                # Split but keep delimiter at end of each part (except last)
                splits = re.split(pattern, part)
                if len(splits) > 1:
                    for i, split in enumerate(splits[:-1]):
                        new_parts.append(split + delimiter.strip())
                    new_parts.append(splits[-1])
                else:
                    new_parts.append(part)
            clause_parts = new_parts
        
        # Convert clause parts to TextUnits with proper char offsets
        current_pos = sent_start
        for clause in clause_parts:
            clause = clause.strip()
            if not clause:
                continue
            
            # Find this clause in the original text starting from current_pos
            clause_offset = text.find(clause, current_pos, sent_end)
            if clause_offset == -1:
                # Fallback: use current_pos
                clause_offset = current_pos
            
            clause_start = clause_offset
            clause_end = clause_offset + len(clause)
            clause_text = text[clause_start:clause_end]
            
            units.append(TextUnit(
                start_char=clause_start,
                end_char=clause_end,
                text=clause_text,
                token_len=len(word_tokenize(clause_text))
            ))
            
            current_pos = clause_end
    
    return units


def compute_span_length_quantiles(spans_df: pd.DataFrame, 
                                   quantiles: List[float] = None) -> List[int]:
    """
    Compute target span lengths from annotated span distribution using quantiles.
    
    Args:
        spans_df: DataFrame with span annotations (must have 'span_text' column)
        quantiles: List of quantiles to compute (default: left-skewed grid)
        
    Returns:
        Sorted unique list of integer span lengths
    """
    if quantiles is None:
        # Left-skewed quantile grid (more short lengths)
        quantiles = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
    
    # Compute span lengths from annotations
    span_lengths = spans_df['span_text'].apply(lambda x: len(word_tokenize(x)))
    
    # Get quantile values
    target_lengths = span_lengths.quantile(quantiles).astype(int).unique().tolist()
    target_lengths = sorted([L for L in target_lengths if L > 0])
    
    return target_lengths


def compute_deterministic_anchors(n_units: int, paragraph_token_len: int, 
                                   median_span_len: int, 
                                   min_anchors: int = 4, max_anchors: int = 10) -> List[int]:
    """
    Compute deterministic anchor positions spaced across the paragraph.
    
    Args:
        n_units: Number of text units in paragraph
        paragraph_token_len: Total tokens in paragraph
        median_span_len: Median annotated span length (for scaling)
        min_anchors: Minimum number of anchors
        max_anchors: Maximum number of anchors
        
    Returns:
        List of anchor indices (unit positions)
    """
    if n_units == 0:
        return []
    
    # Scale anchors based on paragraph length
    n_anchors = max(min_anchors, min(max_anchors, int(np.ceil(paragraph_token_len / median_span_len))))
    
    # Evenly spaced anchors
    if n_anchors >= n_units:
        return list(range(n_units))
    
    anchors = np.round(np.linspace(0, n_units - 1, n_anchors)).astype(int).tolist()
    return sorted(list(set(anchors)))  # Remove duplicates and sort


def is_well_formed_span(text: str, require_letter_start: bool = True, 
                        allow_end_punct: bool = True) -> bool:
    """
    Quality filter for span text.
    
    Rejects:
    - Empty/whitespace-only
    - Leading punctuation (except quotes)
    - Starts with comma, period, etc.
    
    Args:
        text: Span text to validate
        require_letter_start: Enforce spans start with letter
        allow_end_punct: Allow punctuation at end (currently not enforced)
        
    Returns:
        True if span is well-formed, False otherwise
    """
    text = text.strip()
    
    if not text:
        return False
    
    # Bad start characters
    bad_starts = {',', '.', ';', ':', ')', ']', '}', '!', '?'}
    if text[0] in bad_starts:
        return False
    
    # Leading punctuation pattern (excluding quotes at start)
    if re.match(r'^[^\w\s"\'«]+', text):
        return False
    
    # Require letter start
    if require_letter_start and not re.match(r'^[A-Za-z]', text):
        return False
    
    # Optionally check end punctuation (soft requirement)
    # Don't enforce strictly as some valid spans may not end with punct
    
    return True


def compute_span_overlap_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """
    Compute Intersection-over-Union for two char ranges.
    
    Args:
        span1: (start_char, end_char) tuple for first span
        span2: (start_char, end_char) tuple for second span
        
    Returns:
        IoU score between 0.0 and 1.0
    """
    start1, end1 = span1
    start2, end2 = span2
    
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def sample_spans_for_paragraph(
    text: str,
    target_lengths: List[int],
    median_span_len: int = 16,
    max_spans_per_par: int = 20,
    min_anchors: int = 4,
    max_anchors: int = 10,
    overlap_threshold: float = 0.9,
    require_letter_start: bool = True,
) -> List[Dict]:
    """
    Deterministic span sampler for a paragraph.
    
    Algorithm:
    1. Segment into units (spaCy sentences + clause splitting)
    2. Compute deterministic anchors spaced across paragraph
    3. For each (anchor, target_length) pair:
       - Accumulate units forward from anchor to reach target length
       - Extract span using char offsets (preserves original text)
       - Apply quality filters
    4. Deduplicate by overlap IoU
    5. Return up to max_spans_per_par unique spans
    
    Args:
        text: Paragraph text
        target_lengths: List of target span lengths (in tokens) from quantiles
        median_span_len: Median span length for anchor calculation
        max_spans_per_par: Maximum spans to return
        min_anchors, max_anchors: Anchor count constraints
        overlap_threshold: IoU threshold for deduplication
        require_letter_start: Enforce spans start with letter
        
    Returns:
        List of span dictionaries with keys:
        - span_start_char, span_end_char, span_text, span_token_len, 
          anchor_idx, units_covered
    """
    # Step 1: Segment into units
    units = segment_paragraph_into_units(text)
    
    if not units:
        return []
    
    # Step 2: Compute paragraph stats
    paragraph_token_len = sum(u.token_len for u in units)
    
    # Step 3: Compute deterministic anchors
    anchors = compute_deterministic_anchors(
        n_units=len(units),
        paragraph_token_len=paragraph_token_len,
        median_span_len=median_span_len,
        min_anchors=min_anchors,
        max_anchors=max_anchors
    )
    
    # Step 4: Track coverage to distribute spans
    covered = set()
    candidate_spans = []
    
    # Step 5: Generate spans from each anchor and target length
    for anchor_idx in anchors:
        # Prefer uncovered anchors; shift if needed
        actual_anchor = anchor_idx
        if actual_anchor in covered and len(covered) < len(units):
            # Find nearest uncovered unit
            for offset in range(1, len(units)):
                if anchor_idx + offset < len(units) and (anchor_idx + offset) not in covered:
                    actual_anchor = anchor_idx + offset
                    break
                if anchor_idx - offset >= 0 and (anchor_idx - offset) not in covered:
                    actual_anchor = anchor_idx - offset
                    break
        
        for target_len in target_lengths:
            # Accumulate units forward from anchor
            current_len = 0
            end_unit_idx = actual_anchor
            
            for j in range(actual_anchor, len(units)):
                current_len += units[j].token_len
                if current_len >= target_len:
                    end_unit_idx = j
                    break
                end_unit_idx = j
            
            # Extract span text using char offsets (no re-joining)
            span_start = units[actual_anchor].start_char
            span_end = units[end_unit_idx].end_char
            span_text = text[span_start:span_end].strip()
            
            # Quality filter
            if not is_well_formed_span(span_text, require_letter_start=require_letter_start):
                continue
            
            # Recompute char range after strip (adjust if needed)
            # Find where stripped text actually starts/ends in original
            stripped_start = text.find(span_text, span_start, span_end + 1)
            if stripped_start != -1:
                span_start = stripped_start
                span_end = stripped_start + len(span_text)
            
            span_token_len = len(word_tokenize(span_text))
            units_covered_list = list(range(actual_anchor, end_unit_idx + 1))
            
            candidate_spans.append({
                'span_start_char': span_start,
                'span_end_char': span_end,
                'span_text': span_text,
                'span_token_len': span_token_len,
                'anchor_idx': actual_anchor,
                'units_covered': units_covered_list,
            })
            
            # Mark units as covered
            for u_idx in units_covered_list:
                covered.add(u_idx)
    
    # Step 6: Deduplicate by overlap
    final_spans = []
    seen_ranges = []
    
    for span in candidate_spans:
        span_range = (span['span_start_char'], span['span_end_char'])
        
        # Check overlap with existing spans
        is_duplicate = False
        for existing_range in seen_ranges:
            if compute_span_overlap_iou(span_range, existing_range) > overlap_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_spans.append(span)
            seen_ranges.append(span_range)
        
        # Cap total spans
        if len(final_spans) >= max_spans_per_par:
            break
    
    return final_spans


def build_span_training_dataset(
    train_df: pd.DataFrame,
    spans_df: pd.DataFrame,
    target_lengths: List[int],
    median_span_len: int,
    max_spans_per_par: int = 20,
    negative_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Build span-level training dataset for Stage 1.
    
    Positive examples: Annotated PCL spans from Task2
    Negative examples: Sampled spans from non-PCL paragraphs
    
    Args:
        train_df: Paragraph-level data with Task1 labels
        spans_df: Span annotations with Task2 labels
        target_lengths: Target span lengths from quantile grid
        median_span_len: Median annotated span length
        max_spans_per_par: Max spans to sample per paragraph
        negative_ratio: Ratio of negative to positive spans (1.0 = balanced)
        
    Returns:
        DataFrame with columns: par_id, span_text, span_start_char, 
        span_end_char, label (0/1)
    """
    from tqdm import tqdm
    
    span_data = []
    
    # 1. Positive examples: Annotated PCL spans
    print("Collecting positive spans from annotations...")
    for _, row in tqdm(spans_df.iterrows(), total=len(spans_df), desc="Positive spans"):
        span_data.append({
            'par_id': row['par_id'],
            'span_text': row['span_text'],
            'span_start_char': row['span_start_norm'],
            'span_end_char': row['span_finish_norm'],
            'label': 1,
            'source': 'annotation'
        })
    
    n_positive = len(span_data)
    n_negative_target = int(n_positive * negative_ratio)
    
    # 2. Negative examples: Sample from non-PCL paragraphs
    print(f"\nSampling {n_negative_target} negative spans from non-PCL paragraphs...")
    non_pcl_pars = train_df[train_df['label_bin'] == 0]
    
    negative_spans = []
    for _, row in tqdm(non_pcl_pars.iterrows(), total=len(non_pcl_pars), desc="Negative spans"):
        par_text = row['text']
        par_id = row['par_id']
        
        # Sample spans from this paragraph
        sampled = sample_spans_for_paragraph(
            text=par_text,
            target_lengths=target_lengths,
            median_span_len=median_span_len,
            max_spans_per_par=max_spans_per_par,
        )
        
        for span in sampled:
            negative_spans.append({
                'par_id': par_id,
                'span_text': span['span_text'],
                'span_start_char': span['span_start_char'],
                'span_end_char': span['span_end_char'],
                'label': 0,
                'source': 'sampled_neg'
            })
        
        # Stop when we have enough negatives
        if len(negative_spans) >= n_negative_target:
            break
    
    # Trim to target
    negative_spans = negative_spans[:n_negative_target]
    span_data.extend(negative_spans)
    
    # Convert to DataFrame
    span_train_df = pd.DataFrame(span_data)
    
    print(f"\n✓ Span dataset built:")
    print(f"  Positive (PCL): {n_positive:,}")
    print(f"  Negative (non-PCL): {len(negative_spans):,}")
    print(f"  Total: {len(span_train_df):,}")
    print(f"  Class balance: {span_train_df['label'].value_counts(normalize=True).to_dict()}")
    
    return span_train_df


def build_eval_span_candidates(
    eval_df: pd.DataFrame,
    target_lengths: List[int],
    median_span_len: int,
    max_spans_per_par: int = 20,
) -> pd.DataFrame:
    """
    Build span candidates for evaluation/inference (Stage 4).
    
    Samples spans from ALL paragraphs (PCL and non-PCL) for consistent evaluation.
    At inference time, we'll score these spans and aggregate for paragraph prediction.
    
    Args:
        eval_df: Paragraph-level eval data
        target_lengths: Target span lengths from quantile grid
        median_span_len: Median annotated span length
        max_spans_per_par: Max spans to sample per paragraph
        
    Returns:
        DataFrame with columns: par_id, span_text, span_start_char, 
        span_end_char, paragraph_label (ground truth)
    """
    from tqdm import tqdm
    
    span_data = []
    
    print(f"Generating eval span candidates for {len(eval_df)} paragraphs...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Eval spans"):
        par_text = row['text']
        par_id = row['par_id']
        par_label = row['label_bin']
        
        # Sample spans from this paragraph
        sampled = sample_spans_for_paragraph(
            text=par_text,
            target_lengths=target_lengths,
            median_span_len=median_span_len,
            max_spans_per_par=max_spans_per_par,
        )
        
        for span in sampled:
            span_data.append({
                'par_id': par_id,
                'span_text': span['span_text'],
                'span_start_char': span['span_start_char'],
                'span_end_char': span['span_end_char'],
                'paragraph_label': par_label,
            })
    
    eval_spans_df = pd.DataFrame(span_data)
    
    print(f"\n✓ Eval span candidates built:")
    print(f"  Total spans: {len(eval_spans_df):,}")
    print(f"  Avg spans per paragraph: {len(eval_spans_df) / len(eval_df):.1f}")
    print(f"  Paragraphs: PCL={sum(eval_df['label_bin']==1)}, non-PCL={sum(eval_df['label_bin']==0)}")
    
    return eval_spans_df
