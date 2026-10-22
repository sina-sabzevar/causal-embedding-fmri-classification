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
from .causal import METHODS as CAUSAL_METHODS, estimate_graph, estimate_graphs, load_matrices, orient
from .embedding import Node2VecConfig, connectivity_profile, embed_subjects, node2vec_embedding
from .mrmr import MRMRSelector
from .resampling import make_samplers
from .synthetic import make_synthetic_embeddings, make_synthetic_timeseries

__all__ = [
    "Dataset", "load_combined", "load_groups", "subsample_controls",
    "RESAMPLING_OPTIONS", "EvalConfig", "benjamini_hochberg", "build_pipeline",
    "evaluate_all_regions", "evaluate_region", "out_of_fold_scores", "permutation_pvalue",
    "MRMRSelector", "make_synthetic_embeddings",
    "CAUSAL_METHODS", "estimate_graph", "estimate_graphs", "load_matrices", "orient",
    "Node2VecConfig", "connectivity_profile", "embed_subjects", "node2vec_embedding", "make_samplers",
    "make_synthetic_timeseries",
]
