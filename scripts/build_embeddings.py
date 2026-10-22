"""Time series -> causal connectivity graphs -> node2vec embeddings.

Run once per group. The embedding file it writes is the input to
run_classification.py (--controls / --patients).

Examples
--------
From time series, one array (subjects, timepoints, regions):
    python scripts/build_embeddings.py --timeseries data/controls_ts.npy \
        --method lingam --out data/controls

From a folder with one (timepoints, regions) .npy file per subject:
    python scripts/build_embeddings.py --timeseries-dir data/controls_ts/ \
        --method pc --out data/controls

From precomputed matrices, e.g. rDCM exported from MATLAB (skips discovery):
    python scripts/build_embeddings.py --matrices data/rdcm_controls.mat \
        --transpose --out data/controls

Synthetic demo:
    python scripts/build_embeddings.py --synthetic --out results/demo/synthetic

Writes <out>_graphs.npy (subjects, regions, regions) and
<out>_embeddings.npy (subjects, regions, dim).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_embed_clf import (  # noqa: E402
    CAUSAL_METHODS, Node2VecConfig, connectivity_profile, embed_subjects, estimate_graphs, load_matrices,
    make_synthetic_timeseries, orient,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input (choose one)")
    src.add_argument("--timeseries", help=".npy, shape (subjects, timepoints, regions)")
    src.add_argument("--timeseries-dir", help="folder of per-subject .npy files (timepoints, regions)")
    src.add_argument("--matrices", help="precomputed connectivity (.npy or .mat), skips causal discovery")
    src.add_argument("--synthetic", action="store_true")

    p.add_argument("--method", choices=CAUSAL_METHODS, default="lingam")
    p.add_argument("--alpha", type=float, default=0.05, help="PC significance level")
    p.add_argument("--weights", choices=["ols", "binary"], default="ols", help="edge weights for PC/GES")
    p.add_argument("--mat-key", help="variable name inside a .mat file")
    p.add_argument("--transpose", action="store_true",
                   help="matrices are target-by-source (e.g. rDCM A[i,j] = j -> i)")

    p.add_argument("--features", choices=["node2vec", "profile"], default="node2vec",
                   help="node2vec embeddings, or each region's in/out connectivity profile")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--walk-length", type=int, default=20)
    p.add_argument("--walks-per-node", type=int, default=50)
    p.add_argument("--p", type=float, default=2.0)
    p.add_argument("--q", type=float, default=0.5)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", required=True, help="output prefix, e.g. data/controls")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.matrices:
        graphs = orient(load_matrices(a.matrices, a.mat_key), a.transpose)
        print(f"Loaded {graphs.shape[0]} matrices of {graphs.shape[1]} regions")
    else:
        if a.synthetic:
            ts = make_synthetic_timeseries(n_subjects=6, n_regions=8, seed=a.seed)
        elif a.timeseries:
            ts = list(np.load(a.timeseries))
        elif a.timeseries_dir:
            files = sorted(Path(a.timeseries_dir).glob("*.npy"))
            if not files:
                sys.exit(f"No .npy files in {a.timeseries_dir}")
            ts = [np.load(f) for f in files]
            print("Subject order:", [f.stem for f in files])
        else:
            sys.exit("Give --timeseries, --timeseries-dir, --matrices or --synthetic")
        print(f"Estimating {a.method} graphs for {len(ts)} subjects, {ts[0].shape[1]} regions")
        graphs = estimate_graphs(ts, a.method, n_jobs=a.n_jobs, alpha=a.alpha,
                                 weights=a.weights, random_state=a.seed)

    if a.features == "profile":
        emb = connectivity_profile(graphs)
        print("features: connectivity profile (out- and in-edges per region)")
    else:
        cfg = Node2VecConfig(dim=a.dim, walk_length=a.walk_length, walks_per_node=a.walks_per_node,
                             p=a.p, q=a.q, window=a.window, seed=a.seed)
        print(f"features: node2vec, dim={cfg.dim}, p={cfg.p}, q={cfg.q}, shared seed={cfg.seed}")
        emb = embed_subjects(graphs, cfg, n_jobs=a.n_jobs)

    np.save(f"{out}_graphs.npy", graphs)
    np.save(f"{out}_embeddings.npy", emb)
    Path(f"{out}_settings.json").write_text(json.dumps(vars(a), indent=2))
    print(f"Wrote {out}_graphs.npy {graphs.shape} and {out}_embeddings.npy {emb.shape}")


if __name__ == "__main__":
    main()
