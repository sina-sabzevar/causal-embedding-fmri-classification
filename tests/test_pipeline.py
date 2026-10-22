import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_embed_clf import (  # noqa: E402
    EvalConfig, MRMRSelector, benjamini_hochberg, build_pipeline, evaluate_region,
    make_synthetic_embeddings, out_of_fold_scores, subsample_controls,
)
from causal_embed_clf import evaluate as ev  # noqa: E402


def test_mrmr_picks_informative_features_first():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40, int), np.ones(40, int)]
    X = rng.normal(size=(80, 20))
    X[y == 1, 4] += 2.0
    sel = MRMRSelector(k=3).fit(X, y)
    assert sel.selected_features_[0] == 4
    assert sel.transform(X).shape == (80, 3)


def test_feature_selection_never_sees_held_out_subject(monkeypatch):
    """The selector must be fitted inside each fold, on n-1 subjects under LOO."""
    seen_sizes = []
    original_fit = MRMRSelector.fit

    def spy_fit(self, X, y):
        seen_sizes.append(len(X))
        return original_fit(self, X, y)

    monkeypatch.setattr(MRMRSelector, "fit", spy_fit)
    data = make_synthetic_embeddings(n_controls=20, n_patients=10, n_regions=2, dim=16)
    cfg = EvalConfig(k_features=5, resampling="none", cv="loo")
    out_of_fold_scores(data.X[:, 0, :], data.y, cfg)
    n = len(data.y)
    assert seen_sizes and all(s == n - 1 for s in seen_sizes)


@pytest.mark.parametrize("resampling", ev.RESAMPLING_OPTIONS)
def test_every_resampling_option_runs(resampling):
    data = make_synthetic_embeddings(n_controls=40, n_patients=15, n_regions=1, dim=16, informative_regions=(0,))
    cfg = EvalConfig(k_features=8, resampling=resampling, cv="repeated-kfold", n_splits=3, n_repeats=1)
    m = evaluate_region(data.X[:, 0, :], data.y, cfg)
    assert 0.0 <= m["auc"] <= 1.0


def test_signal_is_found_and_noise_is_not():
    data = make_synthetic_embeddings(n_controls=40, n_patients=20, n_regions=6, dim=32,
                                     informative_regions=(2,), effect=1.5, seed=1)
    cfg = EvalConfig(k_features=8, resampling="smote", cv="repeated-kfold", n_splits=5, n_repeats=2)
    informative = evaluate_region(data.X[:, 2, :], data.y, cfg)["auc"]
    noise = evaluate_region(data.X[:, 0, :], data.y, cfg)["auc"]
    assert informative > 0.85
    assert noise < 0.75


def test_resampling_only_on_training_fold():
    """imbalanced-learn pipelines must not resample at prediction time."""
    data = make_synthetic_embeddings(n_controls=30, n_patients=10, n_regions=1, dim=8)
    pipe = build_pipeline(EvalConfig(k_features=4, resampling="smote"))
    X = data.X[:, 0, :]
    pipe.fit(X, data.y)
    assert pipe.decision_function(X).shape == (len(X),)


def test_subsample_controls_keeps_all_patients():
    data = make_synthetic_embeddings(n_controls=50, n_patients=12)
    d = subsample_controls(data, 20, seed=3)
    assert (d.y == 0).sum() == 20 and (d.y == 1).sum() == 12


def test_benjamini_hochberg_matches_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = benjamini_hochberg(p)
    np.testing.assert_allclose(adj, [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)
