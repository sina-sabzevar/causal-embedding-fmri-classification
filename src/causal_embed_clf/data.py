"""Loading node-embedding arrays and building labels.

Expected input: one array per group with shape
``(n_subjects, n_regions, embedding_dim)``, for example node2vec
embeddings of each subject's causal connectivity graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray        # (n_subjects, n_regions, embedding_dim)
    y: np.ndarray        # (n_subjects,), 0 = control, 1 = patient
    groups: tuple[str, str] = ("control", "patient")

    @property
    def n_regions(self) -> int:
        return self.X.shape[1]


def load_groups(control_path: str, patient_path: str) -> Dataset:
    """Load one embedding array per group and stack them with labels."""
    controls = np.load(control_path)
    patients = np.load(patient_path)
    _check_shape(controls, control_path)
    _check_shape(patients, patient_path)
    if controls.shape[1:] != patients.shape[1:]:
        raise ValueError(
            f"Group arrays disagree on (regions, dim): {controls.shape[1:]} vs {patients.shape[1:]}"
        )
    X = np.concatenate([controls, patients], axis=0)
    y = np.concatenate([np.zeros(len(controls), int), np.ones(len(patients), int)])
    return Dataset(X=X, y=y)


def load_combined(embedding_path: str, label_path: str) -> Dataset:
    """Load a single embedding array and a separate label vector (0/1)."""
    X = np.load(embedding_path)
    y = np.load(label_path).astype(int).ravel()
    _check_shape(X, embedding_path)
    if len(X) != len(y):
        raise ValueError(f"{len(X)} subjects but {len(y)} labels")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Labels must be 0 (control) and 1 (patient)")
    return Dataset(X=X, y=y)


def subsample_controls(data: Dataset, n_controls: int, seed: int) -> Dataset:
    """Randomly keep ``n_controls`` controls and all patients.

    Use several seeds and report the spread: a single subsample is
    arbitrary and can move results noticeably in small cohorts.
    """
    rng = np.random.default_rng(seed)
    control_idx = np.flatnonzero(data.y == 0)
    patient_idx = np.flatnonzero(data.y == 1)
    if n_controls > len(control_idx):
        raise ValueError(f"Requested {n_controls} controls, only {len(control_idx)} available")
    keep = np.concatenate([rng.choice(control_idx, n_controls, replace=False), patient_idx])
    keep.sort()
    return Dataset(X=data.X[keep], y=data.y[keep], groups=data.groups)


def _check_shape(arr: np.ndarray, path: str) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (subjects, regions, dim), got shape {arr.shape}")
