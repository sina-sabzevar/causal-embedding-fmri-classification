"""Minimum-redundancy maximum-relevance (mRMR) feature selection.

A small, dependency-free implementation of the F-statistic / correlation
variant of mRMR (relevance = ANOVA F, redundancy = mean absolute Pearson
correlation with the already selected features). This is the same scoring
scheme used by default in the ``mrmr_selection`` package for classification.

The selector is a scikit-learn transformer, so it can sit inside a
cross-validation pipeline. That is the point: features must be chosen
on the training fold only, never on the full dataset.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Select ``k`` features by mRMR (F-statistic relevance, correlation redundancy).

    Parameters
    ----------
    k : int
        Number of features to keep. If larger than the number of features,
        all features are kept.
    """

    def __init__(self, k: int = 30):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        k = min(self.k, n_features)

        with warnings.catch_warnings():          # constant columns give 0/0; scored as 0 below
            warnings.simplefilter("ignore", RuntimeWarning)
            relevance, _ = f_classif(X, y)
        relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0)

        # Absolute Pearson correlation between features, computed once.
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)

        selected: list[int] = [int(np.argmax(relevance))]
        remaining = set(range(n_features)) - set(selected)

        while len(selected) < k and remaining:
            candidates = np.array(sorted(remaining))
            redundancy = corr[np.ix_(candidates, selected)].mean(axis=1)
            score = relevance[candidates] / np.maximum(redundancy, 1e-12)
            best = int(candidates[np.argmax(score)])
            selected.append(best)
            remaining.remove(best)

        self.selected_features_ = np.array(selected, dtype=int)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        X = check_array(X)
        return X[:, self.selected_features_]
