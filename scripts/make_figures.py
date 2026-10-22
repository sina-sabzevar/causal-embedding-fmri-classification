"""Figures and BrainNet Viewer exports.

Subcommands
-----------
heatmap      square connectivity matrix -> heatmap
connections  square connectivity matrix + selected regions -> circular plot
brainnet     region table (CSV) -> BrainNet .node file; optional matrix -> .mat edges
resampling   one region's embeddings -> NearMiss / SVM-SMOTE / Tomek illustration

Examples
--------
python scripts/make_figures.py heatmap --matrix data/subject5_rdcm.npy --out fig/heatmap.png
python scripts/make_figures.py connections --matrix data/subject5_rdcm.npy --regions 5 25 50 --out fig/conn.png
python scripts/make_figures.py brainnet --nodes-csv data/top_regions.csv --out fig/regions.node
python scripts/make_figures.py resampling --synthetic --region 3 --out fig/resampling.png

The --matrix file may be a single (regions, regions) matrix or a stack
(subjects, regions, regions); use --subject to pick one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_embed_clf import MRMRSelector, load_groups, make_synthetic_embeddings  # noqa: E402
from causal_embed_clf.viz import (  # noqa: E402
    keep_rows_for_brainnet, plot_connectivity_heatmap, plot_region_connections,
    plot_resampling_illustration, write_brainnet_nodes,
)


def load_matrix(path: str, subject: int) -> np.ndarray:
    m = np.load(path)
    if m.ndim == 3:
        m = m[subject]
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        sys.exit(f"{path}: need a square (regions, regions) matrix, got {m.shape}. "
                 "Node-embedding arrays (regions, dim) are not connectivity matrices.")
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("heatmap")
    h.add_argument("--matrix", required=True)
    h.add_argument("--subject", type=int, default=0)
    h.add_argument("--highlight", type=int, nargs="*")
    h.add_argument("--out", required=True)

    c = sub.add_parser("connections")
    c.add_argument("--matrix", required=True)
    c.add_argument("--subject", type=int, default=0)
    c.add_argument("--regions", type=int, nargs="+", required=True)
    c.add_argument("--threshold", type=float, default=0.1)
    c.add_argument("--out", required=True)

    b = sub.add_parser("brainnet")
    b.add_argument("--nodes-csv", required=True, help="columns: x,y,z,size,colour,label")
    b.add_argument("--matrix", help="optional connectivity matrix to export as edges (.mat)")
    b.add_argument("--subject", type=int, default=0)
    b.add_argument("--keep-rows", type=int, nargs="*", help="rows (regions) to keep in the edge file")
    b.add_argument("--out", required=True, help=".node output path")

    r = sub.add_parser("resampling")
    r.add_argument("--controls")
    r.add_argument("--patients")
    r.add_argument("--synthetic", action="store_true")
    r.add_argument("--region", type=int, required=True)
    r.add_argument("--k-features", type=int, default=30)
    r.add_argument("--out", required=True)

    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if a.cmd == "heatmap":
        plot_connectivity_heatmap(load_matrix(a.matrix, a.subject), a.out, highlight=a.highlight)
    elif a.cmd == "connections":
        plot_region_connections(load_matrix(a.matrix, a.subject), a.regions, a.out, threshold=a.threshold)
    elif a.cmd == "brainnet":
        df = pd.read_csv(a.nodes_csv)
        write_brainnet_nodes(df[["x", "y", "z", "size", "colour", "label"]].itertuples(index=False), a.out)
        if a.matrix:
            keep_rows_for_brainnet(load_matrix(a.matrix, a.subject), a.keep_rows or [],
                                   Path(a.out).with_suffix(".mat"))
    elif a.cmd == "resampling":
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        data = make_synthetic_embeddings() if a.synthetic else load_groups(a.controls, a.patients)
        X = data.X[:, a.region, :]
        X = MRMRSelector(k=a.k_features).fit_transform(X, data.y)
        X2d = TSNE(n_components=2, random_state=0,
                   perplexity=min(30, len(X) - 1)).fit_transform(StandardScaler().fit_transform(X))
        plot_resampling_illustration(X2d, data.y, a.out)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
