"""
Data processing and sampling utilities for PCL detection.
"""

from .span_sampler import (
    word_tokenize,
    TextUnit,
    SpanAnnotation,
    segment_paragraph_into_units,
    compute_span_length_quantiles,
    compute_deterministic_anchors,
    is_well_formed_span,
    compute_span_overlap_iou,
    sample_spans_for_paragraph,
    build_span_training_dataset,
    build_eval_span_candidates,
)

__all__ = [
    'word_tokenize',
    'TextUnit',
    'SpanAnnotation',
    'segment_paragraph_into_units',
    'compute_span_length_quantiles',
    'compute_deterministic_anchors',
    'is_well_formed_span',
    'compute_span_overlap_iou',
    'sample_spans_for_paragraph',
    'build_span_training_dataset',
    'build_eval_span_candidates',
]
