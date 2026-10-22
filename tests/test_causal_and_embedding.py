import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_embed_clf import (  # noqa: E402
    Node2VecConfig, estimate_graph, load_matrices, make_samplers, make_synthetic_timeseries,
    node2vec_embedding, orient,
)
from causal_embed_clf.causal import _cpdag_to_adjacency  # noqa: E402


def test_lingam_recovers_chain_direction():
    ts = make_synthetic_timeseries(n_subjects=1, n_regions=4, n_timepoints=2000, seed=0)[0]
    A = estimate_graph(ts, "lingam")
    # true edges 0->1, 1->2, 2->3 must be found in the right direction
    for i in range(3):
        assert abs(A[i, i + 1]) > 0.3, f"missing {i}->{i + 1}"
        assert abs(A[i + 1, i]) < 0.1, f"reversed {i + 1}->{i}"


@pytest.mark.parametrize("method", ["pc", "ges"])
def test_pc_and_ges_find_chain_skeleton(method):
    ts = make_synthetic_timeseries(n_subjects=1, n_regions=4, n_timepoints=1000, seed=1)[0]
    A = estimate_graph(ts, method, weights="binary")
    skeleton = (A + A.T) > 0
    for i in range(3):
        assert skeleton[i, i + 1], f"{method} missed edge {i}-{i + 1}"
    assert not skeleton[0, 3]


def test_cpdag_conversion_convention():
    # causal-learn: g[i,j] == -1 and g[j,i] == 1  means i -> j
    g = np.array([[0, -1, -1],
                  [1, 0, 0],
                  [-1, 0, 0]])
    A = _cpdag_to_adjacency(g)
    assert A[0, 1] == 1 and A[1, 0] == 0          # directed 0 -> 1
    assert A[0, 2] == 1 and A[2, 0] == 1          # undirected 0 -- 2 kept both ways


def test_node2vec_shape_and_determinism():
    W = np.zeros((6, 6))
    for i in range(5):
        W[i, i + 1] = 1.0
    cfg = Node2VecConfig(dim=8, walks_per_node=10, walk_length=6, seed=3)
    e1 = node2vec_embedding(W, cfg)
    e2 = node2vec_embedding(W, cfg)
    assert e1.shape == (6, 8)
    np.testing.assert_allclose(e1, e2)


@pytest.mark.parametrize("backend", ["pecanpy", "builtin"])
def test_node2vec_separates_communities(backend):
    # two densely connected groups of 6 regions with no edges between them
    W = np.zeros((12, 12))
    for block in (range(0, 6), range(6, 12)):
        for i in block:
            for j in block:
                if i != j:
                    W[i, j] = 1.0
    emb = node2vec_embedding(W, Node2VecConfig(dim=16, walks_per_node=50, walk_length=10, seed=0, backend=backend))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    S = emb @ emb.T
    same = np.r_[S[:6, :6][~np.eye(6, dtype=bool)], S[6:, 6:][~np.eye(6, dtype=bool)]].mean()
    assert same > S[:6, 6:].mean()


def test_load_matrices_and_orient(tmp_path):
    from scipy.io import savemat

    mats = np.random.default_rng(0).random((5, 5, 3))       # MATLAB style (R, R, subjects)
    savemat(tmp_path / "m.mat", {"A": mats})
    loaded = load_matrices(str(tmp_path / "m.mat"))
    assert loaded.shape == (3, 5, 5)
    np.testing.assert_allclose(orient(loaded, True)[0], loaded[0].T)


def test_make_samplers_rejects_unknown():
    with pytest.raises(ValueError):
        make_samplers("magic")


def test_pecanpy_backend_shape_and_determinism():
    W = np.zeros((6, 6))
    for i in range(5):
        W[i, i + 1] = 0.5
    W[5, 0] = -0.4
    cfg = Node2VecConfig(dim=8, walks_per_node=5, walk_length=6, backend="pecanpy")
    a, b = node2vec_embedding(W, cfg), node2vec_embedding(W, cfg)
    assert a.shape == (6, 8)
    assert np.allclose(a, b)


def test_edge_weight_modes():
    from causal_embed_clf.embedding import _walk_matrix

    W = np.array([[0.0, 0.8, -0.5], [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    assert _walk_matrix(W, "abs")[0, 2] == 0.5
    assert _walk_matrix(W, "positive")[0, 2] == 0.0
