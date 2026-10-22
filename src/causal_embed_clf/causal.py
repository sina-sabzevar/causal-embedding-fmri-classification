"""Causal discovery on regional fMRI time series.

Each subject's time series (timepoints x regions) is turned into a directed,
weighted region x region matrix ``A`` with the convention

    A[i, j] != 0   means an estimated influence  region i -> region j.

Methods (all from causal-learn):

* ``pc``     constraint-based (PC, Fisher-z test). Returns a CPDAG: edges whose
             direction cannot be determined are kept in both directions.
* ``ges``    score-based (GES, BIC score). Same CPDAG handling as PC.
* ``lingam`` functional (DirectLiNGAM). Assumes non-Gaussian noise; returns a
             fully directed, weighted DAG.

PC and GES give edges without strengths. With ``weights="ols"`` each edge is
weighted by the least-squares coefficient of the parent when the child is
regressed on all its parents; with ``weights="binary"`` edges are 1.

rDCM (regression dynamic causal modelling) is not implemented here; it runs in
MATLAB through the TAPAS toolbox. Its connectivity matrices can be loaded with
:func:`load_matrices` and used exactly like the outputs below.
"""

from __future__ import annotations

import numpy as np

METHODS = ("pc", "ges", "lingam")


def _standardise(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, float)
    sd = ts.std(axis=0)
    sd[sd == 0] = 1.0
    return (ts - ts.mean(axis=0)) / sd


def _cpdag_to_adjacency(graph: np.ndarray) -> np.ndarray:
    """Convert causal-learn's endpoint encoding to a directed 0/1 matrix.

    causal-learn: graph[i, j] == -1 and graph[j, i] == 1  means  i -> j;
                  graph[i, j] == graph[j, i] == -1        means  i -- j (undirected).
    """
    g = np.asarray(graph)
    tail = g == -1
    arrow_back = g.T == 1
    directed = tail & arrow_back                     # i -> j
    undirected = tail & (g.T == -1)                  # i -- j, keep both directions
    return (directed | undirected).astype(float)


def _ols_weights(ts: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Weight each edge i -> j by the OLS coefficient of i when j is regressed on its parents."""
    W = np.zeros_like(adjacency, dtype=float)
    for j in range(adjacency.shape[1]):
        parents = np.flatnonzero(adjacency[:, j])
        if parents.size == 0:
            continue
        coef, *_ = np.linalg.lstsq(ts[:, parents], ts[:, j], rcond=None)
        W[parents, j] = coef
    return W


def estimate_graph(ts: np.ndarray, method: str = "lingam", alpha: float = 0.05,
                   weights: str = "ols", random_state: int | None = 0) -> np.ndarray:
    """Estimate a directed connectivity matrix from one subject's time series.

    Parameters
    ----------
    ts : array (timepoints, regions)
    method : "pc", "ges" or "lingam"
    alpha : significance level for PC's conditional independence tests
    weights : "ols" or "binary" (PC and GES only; LiNGAM is always weighted)
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    ts = _standardise(ts)

    if method == "lingam":
        from causallearn.search.FCMBased import lingam

        model = lingam.DirectLiNGAM(random_state=random_state)
        model.fit(ts)
        # causal-learn: B[i, j] is the effect of x_j on x_i. Transpose to i -> j.
        return np.asarray(model.adjacency_matrix_, float).T

    if method == "pc":
        from causallearn.search.ConstraintBased.PC import pc

        graph = pc(ts, alpha, "fisherz", show_progress=False).G.graph
    else:  # ges
        from causallearn.search.ScoreBased.GES import ges

        graph = ges(ts, score_func="local_score_BIC")["G"].graph

    A = _cpdag_to_adjacency(graph)
    if weights == "binary":
        return A
    if weights == "ols":
        return _ols_weights(ts, A)
    raise ValueError("weights must be 'ols' or 'binary'")


def estimate_graphs(timeseries: list[np.ndarray] | np.ndarray, method: str = "lingam",
                    n_jobs: int = 1, **kwargs) -> np.ndarray:
    """Estimate one matrix per subject. Returns (subjects, regions, regions)."""
    from joblib import Parallel, delayed

    mats = Parallel(n_jobs=n_jobs)(delayed(estimate_graph)(ts, method, **kwargs) for ts in timeseries)
    return np.stack(mats)


def load_matrices(path: str, key: str | None = None) -> np.ndarray:
    """Load connectivity matrices from .npy or MATLAB .mat (e.g. rDCM output).

    MATLAB arrays are often stored as (regions, regions, subjects); this is
    converted to (subjects, regions, regions). Check the direction convention
    of your source: rDCM's A[i, j] is usually the influence of region j on i,
    in which case pass ``transpose=True`` to :func:`orient`.
    """
    if path.endswith(".npy"):
        return np.load(path)
    from scipy.io import loadmat

    m = loadmat(path)
    if key is None:
        keys = [k for k in m if not k.startswith("__")]
        if len(keys) != 1:
            raise ValueError(f"{path} has variables {keys}; pass key=...")
        key = keys[0]
    arr = np.asarray(m[key], float)
    if arr.ndim == 3 and arr.shape[0] == arr.shape[1] != arr.shape[2]:
        arr = np.moveaxis(arr, 2, 0)
    return arr


def orient(matrices: np.ndarray, transpose: bool) -> np.ndarray:
    """Transpose each matrix if the source uses target-by-source ordering."""
    return np.swapaxes(matrices, -1, -2) if transpose else matrices
