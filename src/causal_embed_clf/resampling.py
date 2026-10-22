"""Resampling strategies for class imbalance.

Every strategy returns imbalanced-learn sampler steps that are placed inside
the classification pipeline, so they act on the training fold only and never
on held-out subjects.

* ``none``                    no resampling (class weights in the SVM only)
* ``smote``                   SMOTE: interpolate new minority samples
* ``svmsmote``                SVM-SMOTE: generate near the SVM decision boundary
* ``smotetomek``              SMOTE, then remove Tomek links to clean the border
* ``nearmiss-svmsmote-tomek`` undersample the majority with NearMiss-2, oversample
                              with SVM-SMOTE, then remove Tomek links
"""

from __future__ import annotations

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.under_sampling import NearMiss, TomekLinks

RESAMPLING_OPTIONS = ("none", "smote", "svmsmote", "smotetomek", "nearmiss-svmsmote-tomek")


def make_samplers(name: str, random_state: int = 0, nearmiss_ratio: float = 0.6) -> list:
    """Return (name, sampler) pipeline steps for the chosen strategy."""
    if name == "none":
        return []
    if name == "smote":
        return [("smote", SMOTE(random_state=random_state))]
    if name == "svmsmote":
        return [("svmsmote", SVMSMOTE(random_state=random_state))]
    if name == "smotetomek":
        return [("smotetomek", SMOTETomek(random_state=random_state))]
    if name == "nearmiss-svmsmote-tomek":
        return [
            ("nearmiss", NearMiss(version=2, sampling_strategy=nearmiss_ratio)),
            ("svmsmote", SVMSMOTE(random_state=random_state)),
            ("tomek", TomekLinks()),
        ]
    raise ValueError(f"Unknown resampling '{name}'. Options: {RESAMPLING_OPTIONS}")
