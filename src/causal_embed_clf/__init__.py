"""Per-region classification of causal connectivity embeddings from fMRI."""

from .data import Dataset, load_combined, load_groups, subsample_controls
from .evaluate import (
    RESAMPLING_OPTIONS,
    EvalConfig,
    benjamini_hochberg,
    build_pipeline,
    evaluate_all_regions,
    evaluate_region,
    out_of_fold_scores,
    permutation_pvalue,
)
from .mrmr import MRMRSelector
from .synthetic import make_synthetic_embeddings

__all__ = [
    "Dataset", "load_combined", "load_groups", "subsample_controls",
    "RESAMPLING_OPTIONS", "EvalConfig", "benjamini_hochberg", "build_pipeline",
    "evaluate_all_regions", "evaluate_region", "out_of_fold_scores", "permutation_pvalue",
    "MRMRSelector", "make_synthetic_embeddings",
]
