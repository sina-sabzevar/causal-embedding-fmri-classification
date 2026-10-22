"""node2vec embeddings of directed, weighted connectivity graphs.

Each region becomes a vector summarising its position in the subject's causal
graph. Walks follow edge direction (source -> target) and edge strength.
The return parameter ``p`` and in-out parameter ``q`` bias the walk as in
Grover & Leskovec (2016):

    returning to the previous node      weight / p
    moving to a neighbour of it         weight
    moving further away                 weight / q

With ``p=2, q=0.5`` walks avoid going straight back and tend to explore
outwards, so embeddings reflect longer causal pathways rather than only local
neighbourhoods.

Two backends compute the same thing:

* ``"pecanpy"`` (default): PecanPy (Liu & Krishnan, 2021), the implementation
  used in the paper.
* ``"builtin"``: a short pure-Python version, useful for reading the algorithm
  or when PecanPy cannot be installed.

Edge weights: causal graphs have negative edges, and PecanPy never walks along
a negative edge. ``edge_weights="abs"`` (default) walks on absolute strengths,
so inhibitory and excitatory influences both count. ``edge_weights="positive"``
keeps only positive edges, which is what PecanPy does when it is given a
signed matrix directly.

Important limitation: node2vec is fitted separately for each subject, and the
resulting vector spaces are only defined up to an arbitrary rotation. Using the
same random seed for every subject (the default here) makes the spaces partly
comparable, but not guaranteed to be. In synthetic tests, giving each subject a
different seed removes the group difference entirely. Check region-level
findings against :func:`connectivity_profile`, whose features are comparable
across subjects by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node2VecConfig:
    dim: int = 64
    walk_length: int = 20
    walks_per_node: int = 50
    p: float = 2.0
    q: float = 0.5
    window: int = 5
    epochs: int = 5
    seed: int = 0
    backend: str = "pecanpy"        # "pecanpy" or "builtin"
    edge_weights: str = "abs"       # "abs" or "positive"


BACKENDS = ("pecanpy", "builtin")
EDGE_WEIGHTS = ("abs", "positive")


def _walk_matrix(W: np.ndarray, edge_weights: str) -> np.ndarray:
    W = np.asarray(W, float).copy()
    np.fill_diagonal(W, 0.0)
    if edge_weights == "abs":
        return np.abs(W)
    if edge_weights == "positive":
        return np.clip(W, 0.0, None)
    raise ValueError(f"edge_weights must be one of {EDGE_WEIGHTS}, got {edge_weights!r}")


def _random_walks(W: np.ndarray, cfg: Node2VecConfig, rng: np.random.Generator) -> list[list[str]]:
    n = W.shape[0]
    out_neighbours = [np.flatnonzero(W[i]) for i in range(n)]
    walks = []
    for _ in range(cfg.walks_per_node):
        for start in rng.permutation(n):
            walk = [int(start)]
            while len(walk) < cfg.walk_length:
                cur = walk[-1]
                nbrs = out_neighbours[cur]
                if nbrs.size == 0:
                    break
                w = W[cur, nbrs].copy()
                if len(walk) > 1:
                    prev = walk[-2]
                    linked = (W[prev, nbrs] > 0) | (W[nbrs, prev] > 0)
                    bias = np.where(nbrs == prev, 1.0 / cfg.p, np.where(linked, 1.0, 1.0 / cfg.q))
                    w = w * bias
                walk.append(int(rng.choice(nbrs, p=w / w.sum())))
            walks.append([str(v) for v in walk])
    return walks


def _embed_builtin(W: np.ndarray, cfg: Node2VecConfig) -> np.ndarray:
    from gensim.models import Word2Vec

    rng = np.random.default_rng(cfg.seed)
    walks = _random_walks(W, cfg, rng)
    model = Word2Vec(walks, vector_size=cfg.dim, window=cfg.window, min_count=0, sg=1,
                     workers=1, epochs=cfg.epochs, seed=cfg.seed)
    return np.stack([model.wv[str(i)] for i in range(W.shape[0])])


def _embed_pecanpy(W: np.ndarray, cfg: Node2VecConfig) -> np.ndarray:
    import numba
    from pecanpy import pecanpy

    # PecanPy generates walks on several threads, each seeded separately, and
    # which walks a thread gets depends on scheduling. With more than one thread
    # the same seed gives different embeddings on every run. One thread per
    # process makes it reproducible; parallelise over subjects instead (n_jobs).
    numba.set_num_threads(1)

    ids = [str(i) for i in range(W.shape[0])]
    g = pecanpy.DenseOTF.from_mat(W.astype(np.float32), ids, p=cfg.p, q=cfg.q,
                                  workers=1, random_state=cfg.seed)
    return np.asarray(g.embed(dim=cfg.dim, num_walks=cfg.walks_per_node, walk_length=cfg.walk_length,
                              window_size=cfg.window, epochs=cfg.epochs))


def node2vec_embedding(W: np.ndarray, cfg: Node2VecConfig | None = None) -> np.ndarray:
    """Embed one connectivity matrix, ``W[i, j]`` = edge i -> j. Returns (regions, dim).

    Regions with no edges still get a vector, but it carries no structural
    information.
    """
    cfg = cfg or Node2VecConfig()
    Wp = _walk_matrix(W, cfg.edge_weights)
    if cfg.backend == "pecanpy":
        return _embed_pecanpy(Wp, cfg)
    if cfg.backend == "builtin":
        return _embed_builtin(Wp, cfg)
    raise ValueError(f"backend must be one of {BACKENDS}, got {cfg.backend!r}")


def embed_subjects(matrices: np.ndarray, cfg: Node2VecConfig | None = None, n_jobs: int = 1) -> np.ndarray:
    """Embed every subject's matrix. Returns (subjects, regions, dim)."""
    from joblib import Parallel, delayed

    embs = Parallel(n_jobs=n_jobs)(delayed(node2vec_embedding)(m, cfg) for m in matrices)
    return np.stack(embs)


def connectivity_profile(matrices: np.ndarray) -> np.ndarray:
    """Region features taken directly from the connectivity matrix.

    For each region: its outgoing weights followed by its incoming weights.
    Returns (subjects, regions, 2 * regions).

    Unlike node2vec, these features are in the same coordinate system for every
    subject by construction (feature k always means "edge to/from region k"),
    so they are a useful check on any region-level finding.
    """
    m = np.asarray(matrices, float)
    return np.concatenate([m, np.swapaxes(m, -1, -2)], axis=-1)
