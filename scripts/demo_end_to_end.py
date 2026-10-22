"""End-to-end demo on synthetic data.

time series -> LiNGAM graphs -> region features -> per-region classification

Controls and patients share a causal chain over 16 regions. Patients have one
extra edge, region 2 -> region 4. The demo compares two kinds of region
features: node2vec embeddings and the region's own connectivity profile.

    python scripts/demo_end_to_end.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_embed_clf import (  # noqa: E402
    EvalConfig, Node2VecConfig, connectivity_profile, embed_subjects, estimate_graphs,
    evaluate_all_regions, make_synthetic_timeseries,
)


def main() -> None:
    n_regions = 16
    controls = make_synthetic_timeseries(30, n_regions, 300, seed=1)
    patients = make_synthetic_timeseries(15, n_regions, 300, extra_edges=((2, 4),), seed=2)
    y = np.r_[np.zeros(len(controls), int), np.ones(len(patients), int)]

    print("Estimating LiNGAM graphs ...")
    graphs = estimate_graphs(controls + patients, "lingam")
    cfg = EvalConfig(k_features=8, resampling="smote")

    features = {
        "connectivity profile": connectivity_profile(graphs),
        "node2vec, PecanPy (shared seed)": embed_subjects(
            graphs, Node2VecConfig(dim=16, walks_per_node=30, walk_length=10, seed=0), n_jobs=-1),
    }
    for name, X in features.items():
        table = evaluate_all_regions(X, y, cfg)
        print(f"\nAUC by region, {name}:")
        print(table["auc"].round(2).head(8).to_string())

    print("\nThe only group difference is the edge 2 -> 4. Connectivity-profile features rank the regions "
          "on and downstream of that edge highest. With node2vec, regions unrelated to the edge can score "
          "almost as high, because embedding coordinates are not aligned across subjects.")


if __name__ == "__main__":
    main()
