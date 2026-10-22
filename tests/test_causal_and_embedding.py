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


def test_node2vec_neighbours_are_closer_than_distant_nodes():
    # two separate chains: nodes within a chain should be more similar
    W = np.zeros((8, 8))
    for i in range(3):
        W[i, i + 1] = 1.0
        W[i + 4, i + 5] = 1.0
    emb = node2vec_embedding(W, Node2VecConfig(dim=16, walks_per_node=200, walk_length=10, seed=0))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    within = emb[1] @ emb[2]
    across = emb[1] @ emb[6]
    assert within > across


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
