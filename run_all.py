#!/usr/bin/env python3
"""Reproducible resampling benchmark for User Toy_imbalanced.

This version adds a proper ablation study of the proposed TopoGeoFlow
oversampler. The TopoGeoFlowOversampler class has been made configurable
(use_topology / use_geometry / use_boundary_guard) so that individual
components can be switched on or off without duplicating the class. The
"Full TopoGeoFlow" ablation variant (all three flags True) is mathematically
identical to the original, non-configurable implementation.

Ablation-specific code is marked with "# --- ABLATION ---" comments.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
    SMOTE,
    SVMSMOTE,
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / "Dataset" / "Toy_imbalanced.csv"
DEFAULT_OUTPUT = ROOT / "results" / "Toy_imbalanced"

METHOD_ORDER = [
    "Random Under-Sampling (RUS)",
    "Random Over-Sampling (ROS)",
    "SMOTE (SMT)",
    "Borderline-SMOTE (BS)",
    "ADASYN (Ads)",
    "SVM-SMOTE (SSMT)",
    "SMOTE-ENN (SMTN)",
    "SMOTE-Tomek (SMTT)",
    "SMOTE-RD based on relative and absolute densities (RSMT) [1]",
    "Triangular Six-Line Oversampling (TSMT) [2]",
    "SMOTE-based potato-yield data augmentation (SPYP) [3]",
    "TSP-SMOTE: triangular scoring with random perturbation [4]",
    "SMOTE-Complex [5]",
    "SMOTE-Complex-Clustering [5]",
    "TopoGeoFlow (Proposed)",
]

METHOD_TICK_LABELS = [
    "RUS", "ROS", "SMT", "BS", "Ads", "SSMT", "SMTN", "SMTT",
    "RSMT [1]", "TSMT [2]", "SPYP [3]", "TSP-SMOTE [4]",
    "SMOTE-Complex [5]", "SMOTE-Complex-Clustering [5]",
    "TopoGeoFlow (Proposed)",
]

# --- ABLATION ---
# Y-axis order for the ablation-study figures, exactly as requested: the
# fully stripped-down interpolation-only variant first, progressively adding
# one proposed component at a time, ending with the full proposed method.
ABLATION_ORDER = [
    "Basic interpolation",
    "+ Topology",
    "+ Topology + Geometry",
    "+ Topology + Boundary",
    "+ Geometry + Boundary",
    "Full TopoGeoFlow",
]
# --- END ABLATION ---


def save_csv_safely(frame: pd.DataFrame, path: Path) -> Path:
    """Save CSV, using a timestamped sibling when Excel has locked the target."""
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        frame.to_csv(fallback, index=False)
        print(f"{path.name} is locked; saved {fallback.name} instead.")
        return fallback


def load_keel_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a binary dataset, detecting CSV files and misnamed Excel workbooks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    # XLSX files are ZIP containers. Detect their content rather than trusting a
    # possibly incorrect extension (the supplied abalone-17_vs_7-8-9-10.csv is actually XLSX).
    if zipfile.is_zipfile(path):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except (ImportError, ValueError) as exc:
            raise ValueError(
                f"{path} is an Excel workbook despite its {path.suffix!r} "
                "extension, but it could not be read. Install openpyxl or "
                "resave the workbook as a real CSV file."
            ) from exc
    else:
        # UTF-8 is preferred, while cp1252 covers CSV exports from older
        # Windows/Excel versions. Both attempts use strict decoding.
        decode_errors: list[str] = []
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                decode_errors.append(f"{encoding}: {exc}")
        else:
            raise ValueError(
                f"Could not decode CSV dataset {path}. Tried utf-8-sig and "
                f"cp1252. Details: {'; '.join(decode_errors)}"
            )

    if df.empty or df.shape[1] < 2:
        raise ValueError(
            f"Dataset {path} must contain at least one row, one feature "
            "column, and a final target column."
        )

    # The last column is the target
    target_name = df.columns[-1]
    y = df.pop(target_name)

    if y.isna().any():
        raise ValueError(f"Target column {target_name!r} contains missing values.")

    # Accept the common binary conventions while always returning integer 0/1.
    labels = list(pd.unique(y))
    if len(labels) != 2:
        raise ValueError(
            f"The final column {target_name!r} must be a binary target, but "
            f"it contains {len(labels)} distinct values. The supplied "
            f"SECOM workbook appears to contain only its feature columns; "
            "add the SECOM labels as the final column before running."
        )
    normalized = {str(label).strip().lower(): label for label in labels}
    if set(normalized) == {"negative", "positive"}:
        y = y.map(
            {normalized["negative"]: 0, normalized["positive"]: 1}
        )
    elif set(labels) == {-1, 1}:
        y = y.map({-1: 0, 1: 1})
    elif set(labels) != {0, 1}:
        raise ValueError(
            f"Unsupported target labels in {target_name!r}: {labels!r}. "
            "Use 0/1, -1/1, or negative/positive."
        )

    # Convert numeric columns
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors='coerce')
        if converted.notna().all():
            df[column] = converted

    return df, y.astype(int)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )


class SMOTERD:
    """Reproducible SMOTE-RD adaptation using relative/absolute density.

    Relative density removes unsafe minority points whose local neighborhoods
    are majority dominated. Absolute minority density and boundary proximity
    then weight seed selection so sparse and boundary regions receive more
    synthetic samples. Samples are interpolated between minority neighbours.
    """

    def __init__(self, random_state: int = 42, k_neighbors: int = 7):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        minority = labels[np.argmin(counts)]
        majority = labels[np.argmax(counts)]
        X_min = X[y == minority]
        X_maj = X[y == majority]
        needed = len(X_maj) - len(X_min)
        if needed <= 0:
            return X.copy(), y.copy()

        k = min(self.k_neighbors + 1, len(X_min))
        neighbors = NearestNeighbors(n_neighbors=k).fit(X_min)
        distances, indices = neighbors.kneighbors(X_min)
        local_scale = np.maximum(distances[:, 1:].mean(axis=1), 1e-12)
        all_k = min(self.k_neighbors + 1, len(X))
        all_nn = NearestNeighbors(n_neighbors=all_k).fit(X)
        _, all_indices = all_nn.kneighbors(X_min)
        local_labels = y[all_indices[:, 1:]]
        safe_mask = (local_labels == minority).mean(axis=1) >= 0.5
        safe_indices = np.flatnonzero(safe_mask)
        if len(safe_indices) < 2:
            safe_indices = np.arange(len(X_min))

        majority_nn = NearestNeighbors(n_neighbors=1).fit(X_maj)
        boundary_distance = majority_nn.kneighbors(X_min)[0][:, 0]
        sparsity = local_scale / local_scale.sum()
        boundary = 1.0 / np.maximum(boundary_distance, 1e-12)
        boundary /= boundary.sum()
        weights = sparsity + boundary
        weights[~np.isin(np.arange(len(X_min)), safe_indices)] = 0
        weights /= weights.sum()
        rng = np.random.default_rng(self.random_state)

        synthetic: list[np.ndarray] = []
        while len(synthetic) < needed:
            anchor = int(rng.choice(len(X_min), p=weights))
            neighbor = int(rng.choice(indices[anchor, 1:]))
            lam = float(rng.random())
            candidate = X_min[anchor] + lam * (X_min[neighbor] - X_min[anchor])
            synthetic.append(candidate)

        X_syn = np.vstack(synthetic)
        return (
            np.vstack([X, X_syn]),
            np.concatenate([y, np.full(needed, minority, dtype=y.dtype)]),
        )


class TriangularSixLineOversampler:
    """TSMT adaptation based on triangle medians and a variance regulator.

    For each synthetic point, three nearby minority observations form a
    triangle. A point is drawn on one of its six directed vertex-to-opposite-
    midpoint median lines. A regulator shrinks the point toward the triangle
    centroid to keep its dispersion close to the source minority distribution.
    """

    def __init__(self, random_state: int = 42, k_neighbors: int = 7):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        minority = labels[np.argmin(counts)]
        majority = labels[np.argmax(counts)]
        X_min = X[y == minority]
        needed = int((y == majority).sum() - len(X_min))
        if needed <= 0:
            return X.copy(), y.copy()

        k = min(self.k_neighbors + 1, len(X_min))
        nn = NearestNeighbors(n_neighbors=k).fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)
        rng = np.random.default_rng(self.random_state)
        source_variance = float(np.mean(np.var(X_min, axis=0)))
        generated: list[np.ndarray] = []
        for _ in range(needed):
            anchor = int(rng.integers(len(X_min)))
            pool = indices[anchor, 1:]
            chosen = rng.choice(pool, size=2, replace=len(pool) < 2)
            triangle = X_min[[anchor, int(chosen[0]), int(chosen[1])]]
            vertex_index = int(rng.integers(3))
            vertex = triangle[vertex_index]
            opposite = np.delete(triangle, vertex_index, axis=0).mean(axis=0)
            candidate = vertex + rng.random() * (opposite - vertex)
            centroid = triangle.mean(axis=0)
            triangle_variance = float(np.mean(np.var(triangle, axis=0)))
            regulator = min(
                1.0, np.sqrt(source_variance / max(triangle_variance, 1e-12))
            )
            generated.append(centroid + regulator * (candidate - centroid))
        X_syn = np.vstack(generated)
        return (
            np.vstack([X, X_syn]),
            np.concatenate([y, np.full(needed, minority, dtype=y.dtype)]),
        )


class PotatoYieldSMOTE:
    """SPYP adaptation: standard SMOTE augmentation used in the yield study.

    The original study is a regression augmentation application rather than a
    new classification sampler. Here its SMOTE generation step is applied only
    to each training fold and stops when binary class balance is reached.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return SMOTE(random_state=self.random_state).fit_resample(X, y)


class TSPSMOTE:
    """Reproducible TSP-SMOTE adaptation.

    Local minority triangles are scored by compactness and distance from the
    majority class. Synthetic points are drawn inside high-scoring triangles
    and receive a small, locally scaled random perturbation.
    """

    def __init__(
        self, random_state: int = 42, k_neighbors: int = 7, noise: float = 0.03
    ):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.noise = noise

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        minority = labels[np.argmin(counts)]
        majority = labels[np.argmax(counts)]
        X_min, X_maj = X[y == minority], X[y == majority]
        needed = len(X_maj) - len(X_min)
        if needed <= 0:
            return X.copy(), y.copy()
        if len(X_min) < 3:
            return RandomOverSampler(
                random_state=self.random_state
            ).fit_resample(X, y)

        k = min(self.k_neighbors + 1, len(X_min))
        distances, indices = NearestNeighbors(n_neighbors=k).fit(
            X_min
        ).kneighbors(X_min)
        majority_nn = NearestNeighbors(n_neighbors=1).fit(X_maj)
        boundary_distance = majority_nn.kneighbors(X_min)[0][:, 0]
        local_scale = np.maximum(distances[:, 1:].mean(axis=1), 1e-12)
        scores = boundary_distance / local_scale
        scores = np.maximum(scores, 1e-12)
        scores /= scores.sum()
        rng = np.random.default_rng(self.random_state)
        generated: list[np.ndarray] = []
        for _ in range(needed):
            anchor = int(rng.choice(len(X_min), p=scores))
            pool = indices[anchor, 1:]
            pair = rng.choice(pool, size=2, replace=len(pool) < 2)
            triangle = X_min[[anchor, int(pair[0]), int(pair[1])]]
            weights = rng.dirichlet(np.full(3, 2.0))
            candidate = weights @ triangle
            candidate += rng.normal(
                scale=self.noise * local_scale[anchor], size=X.shape[1]
            )
            generated.append(candidate)
        return (
            np.vstack([X, np.vstack(generated)]),
            np.concatenate([y, np.full(needed, minority, dtype=y.dtype)]),
        )


class SMOTEComplex:
    """Complexity-aware SMOTE adaptation with optional minority clustering."""

    def __init__(
        self,
        random_state: int = 42,
        k_neighbors: int = 7,
        clustering: bool = False,
    ):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.clustering = clustering

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        minority = labels[np.argmin(counts)]
        majority = labels[np.argmax(counts)]
        X_min, X_maj = X[y == minority], X[y == majority]
        needed = len(X_maj) - len(X_min)
        if needed <= 0:
            return X.copy(), y.copy()
        if len(X_min) < 2:
            return RandomOverSampler(
                random_state=self.random_state
            ).fit_resample(X, y)

        k = min(self.k_neighbors + 1, len(X_min))
        distances, indices = NearestNeighbors(n_neighbors=k).fit(
            X_min
        ).kneighbors(X_min)
        all_k = min(self.k_neighbors + 1, len(X))
        local_indices = NearestNeighbors(n_neighbors=all_k).fit(X).kneighbors(
            X_min, return_distance=False
        )[:, 1:]
        majority_fraction = (y[local_indices] == majority).mean(axis=1)
        sparsity = distances[:, 1:].mean(axis=1)
        weights = (0.25 + majority_fraction) * np.maximum(sparsity, 1e-12)

        cluster_labels = np.zeros(len(X_min), dtype=int)
        if self.clustering and len(X_min) >= 6:
            n_clusters = min(5, max(2, int(np.sqrt(len(X_min) / 2))))
            cluster_labels = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=self.random_state
            ).fit_predict(X_min)
            cluster_sizes = np.bincount(cluster_labels)
            weights *= 1.0 / cluster_sizes[cluster_labels]
        weights /= weights.sum()

        rng = np.random.default_rng(self.random_state)
        generated: list[np.ndarray] = []
        for _ in range(needed):
            anchor = int(rng.choice(len(X_min), p=weights))
            pool = indices[anchor, 1:]
            if self.clustering:
                same_cluster = pool[
                    cluster_labels[pool] == cluster_labels[anchor]
                ]
                if len(same_cluster):
                    pool = same_cluster
            neighbor = int(rng.choice(pool))
            lam = float(rng.beta(2.0, 2.0))
            generated.append(
                X_min[anchor] + lam * (X_min[neighbor] - X_min[anchor])
            )
        return (
            np.vstack([X, np.vstack(generated)]),
            np.concatenate([y, np.full(needed, minority, dtype=y.dtype)]),
        )


class TopoGeoFlowOversampler:
    """Benchmark prototype of our topology/geometry-aware oversampler.

    Minority points are connected through a mutual-kNN graph so interpolation
    follows locally supported topology. Candidate points move along a graph
    edge with small locally scaled geometric noise and are accepted only when
    they remain closer to the minority class than to the majority class.

    This lightweight sampler tests the structural generation principle in this
    benchmark; it is not the repository's planned neural conditional-flow model.

    # --- ABLATION ---
    Three proposed components can each be switched on or off independently so
    that a single implementation supports the full ablation study without
    duplicating code:

    - ``use_topology``: restrict candidate neighbours to the *mutual* kNN
      graph (Eq. 1 in the paper). When False, ordinary (possibly asymmetric)
      kNN neighbours are used instead.
    - ``use_geometry``: add a locally scaled perturbation that is orthogonal
      to the interpolation edge. When False, candidates are pure edge
      interpolations with no added noise.
    - ``use_boundary_guard``: reject candidates that are not sufficiently
      closer to the minority support than to the majority support, retrying
      until enough safe candidates are found (falling back to plain edge
      interpolation to still guarantee exact class balance). When False,
      every generated candidate is accepted immediately, so exactly the
      needed number of samples is produced in a single pass.

    With all three flags True this is mathematically identical to the
    original, non-configurable implementation ("Full TopoGeoFlow").
    # --- END ABLATION ---
    """

    def __init__(
        self,
        random_state: int = 42,
        k_neighbors: int = 7,
        geometry_noise: float = 0.05,
        boundary_margin: float = 0.90,
        use_topology: bool = True,
        use_geometry: bool = True,
        use_boundary_guard: bool = True,
    ):
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.geometry_noise = geometry_noise
        self.boundary_margin = boundary_margin
        # --- ABLATION ---
        self.use_topology = use_topology
        self.use_geometry = use_geometry
        self.use_boundary_guard = use_boundary_guard
        # --- END ABLATION ---

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        minority = labels[np.argmin(counts)]
        majority = labels[np.argmax(counts)]
        X_min = X[y == minority]
        X_maj = X[y == majority]
        needed = len(X_maj) - len(X_min)
        if needed <= 0:
            return X.copy(), y.copy()
        if len(X_min) < 2:
            return RandomOverSampler(
                random_state=self.random_state
            ).fit_resample(X, y)

        k = min(self.k_neighbors + 1, len(X_min))
        minority_nn = NearestNeighbors(n_neighbors=k).fit(X_min)
        distances, neighbor_indices = minority_nn.kneighbors(X_min)

        # --- ABLATION: topology component ---
        # When enabled, only keep neighbour pairs whose kNN relationship is
        # reciprocal (mutual-kNN graph, Eq. 1). When disabled, fall back to
        # ordinary (possibly asymmetric) kNN neighbours.
        if self.use_topology:
            neighbor_sets = [
                set(row[1:].tolist()) for row in neighbor_indices
            ]
            candidate_neighbors: list[np.ndarray] = []
            for index, row in enumerate(neighbor_indices):
                mutual = [
                    candidate
                    for candidate in row[1:]
                    if index in neighbor_sets[int(candidate)]
                ]
                if not mutual:
                    mutual = row[1:].tolist()
                candidate_neighbors.append(np.asarray(mutual, dtype=int))
        else:
            candidate_neighbors = [
                row[1:].astype(int) for row in neighbor_indices
            ]
        # --- END ABLATION ---

        local_scale = np.maximum(
            distances[:, 1:].mean(axis=1), np.finfo(float).eps
        )

        # --- ABLATION: boundary-guard component ---
        # The majority/minority-distance structures used by the rejection
        # criterion are only needed when the guard is enabled.
        if self.use_boundary_guard:
            majority_nn = NearestNeighbors(n_neighbors=1).fit(X_maj)
            minority_guard = NearestNeighbors(
                n_neighbors=min(2, len(X_min))
            ).fit(X_min)
        # --- END ABLATION ---

        rng = np.random.default_rng(self.random_state)
        synthetic: list[np.ndarray] = []
        # Without the boundary guard there is no rejection, so a single pass
        # of exactly `needed` draws always produces exactly `needed` samples.
        max_attempts = (
            max(needed * 50, 1000) if self.use_boundary_guard else needed
        )

        for _ in range(max_attempts):
            if len(synthetic) >= needed:
                break
            anchor_index = int(rng.integers(len(X_min)))
            neighbor_index = int(rng.choice(candidate_neighbors[anchor_index]))
            anchor = X_min[anchor_index]
            neighbor = X_min[neighbor_index]

            # Beta interpolation favors the supported interior of graph edges.
            interpolation = float(rng.beta(2.0, 2.0))
            candidate = anchor + interpolation * (neighbor - anchor)

            # --- ABLATION: geometry component ---
            if self.use_geometry:
                noise = rng.normal(
                    scale=self.geometry_noise * local_scale[anchor_index],
                    size=X.shape[1],
                )
                edge = neighbor - anchor
                edge_norm = float(np.dot(edge, edge))
                if edge_norm > np.finfo(float).eps:
                    noise -= edge * (float(np.dot(noise, edge)) / edge_norm)
                candidate = candidate + noise
            # --- END ABLATION ---

            # --- ABLATION: boundary-guard component ---
            if self.use_boundary_guard:
                distance_to_majority = float(
                    majority_nn.kneighbors(candidate.reshape(1, -1))[0][0, 0]
                )
                minority_distances = minority_guard.kneighbors(
                    candidate.reshape(1, -1)
                )[0][0]
                distance_to_minority = float(minority_distances[0])
                if distance_to_majority >= (
                    self.boundary_margin * distance_to_minority
                ):
                    synthetic.append(candidate)
                # else: reject and resample in the next attempt.
            else:
                synthetic.append(candidate)
            # --- END ABLATION ---

        # A graph-edge interpolation fallback guarantees exact class balance
        # (only ever needed when the boundary guard can reject candidates).
        while len(synthetic) < needed:
            anchor_index = int(rng.integers(len(X_min)))
            neighbor_index = int(rng.choice(candidate_neighbors[anchor_index]))
            interpolation = float(rng.beta(2.0, 2.0))
            synthetic.append(
                X_min[anchor_index]
                + interpolation
                * (X_min[neighbor_index] - X_min[anchor_index])
            )

        X_syn = np.vstack(synthetic[:needed])
        return (
            np.vstack([X, X_syn]),
            np.concatenate(
                [y, np.full(needed, minority, dtype=y.dtype)]
            ),
        )


def classifiers(seed: int) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight=None, random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", probability=True, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=-1,
        ),
    }


def samplers(seed: int) -> dict[str, object | None]:
    return {
        "Random Under-Sampling (RUS)": RandomUnderSampler(random_state=seed),
        "Random Over-Sampling (ROS)": RandomOverSampler(random_state=seed),
        "SMOTE (SMT)": SMOTE(random_state=seed),
        "Borderline-SMOTE (BS)": BorderlineSMOTE(random_state=seed),
        "ADASYN (Ads)": ADASYN(random_state=seed),
        "SVM-SMOTE (SSMT)": SVMSMOTE(random_state=seed),
        "SMOTE-ENN (SMTN)": SMOTEENN(random_state=seed),
        "SMOTE-Tomek (SMTT)": SMOTETomek(random_state=seed),
        "SMOTE-RD based on relative and absolute densities (RSMT) [1]": SMOTERD(random_state=seed),
        "Triangular Six-Line Oversampling (TSMT) [2]": TriangularSixLineOversampler(random_state=seed),
        "SMOTE-based potato-yield data augmentation (SPYP) [3]": PotatoYieldSMOTE(random_state=seed),
        "TSP-SMOTE: triangular scoring with random perturbation [4]": TSPSMOTE(random_state=seed),
        "SMOTE-Complex [5]": SMOTEComplex(random_state=seed),
        "SMOTE-Complex-Clustering [5]": SMOTEComplex(
            random_state=seed, clustering=True
        ),
        # Full TopoGeoFlow: all three proposed components enabled (default).
        # This is mathematically identical to the original implementation.
        "TopoGeoFlow (Proposed)": TopoGeoFlowOversampler(random_state=seed),
    }


# --- ABLATION ---
def ablation_variants(seed: int) -> dict[str, object]:
    """TopoGeoFlow ablation variants, one configurable class instantiated six
    different ways so that each proposed component's individual and joint
    contribution can be measured under identical experimental conditions.
    """
    return {
        "Basic interpolation": TopoGeoFlowOversampler(
            random_state=seed,
            use_topology=False,
            use_geometry=False,
            use_boundary_guard=False,
        ),
        "+ Topology": TopoGeoFlowOversampler(
            random_state=seed,
            use_topology=True,
            use_geometry=False,
            use_boundary_guard=False,
        ),
        "+ Topology + Geometry": TopoGeoFlowOversampler(
            random_state=seed,
            use_topology=True,
            use_geometry=True,
            use_boundary_guard=False,
        ),
        "+ Topology + Boundary": TopoGeoFlowOversampler(
            random_state=seed,
            use_topology=True,
            use_geometry=False,
            use_boundary_guard=True,
        ),
        "+ Geometry + Boundary": TopoGeoFlowOversampler(
            random_state=seed,
            use_topology=False,
            use_geometry=True,
            use_boundary_guard=True,
        ),
        "Full TopoGeoFlow": TopoGeoFlowOversampler(
            random_state=seed,
            use_topology=True,
            use_geometry=True,
            use_boundary_guard=True,
        ),
    }
# --- END ABLATION ---


def save_comparison_figure(raw: pd.DataFrame, output_dir: Path) -> Path:
    """Create a publication-ready four-metric comparison in the stated order."""
    metrics = ["AUC", "F1", "G-Mean", "Accuracy"]
    panel_titles = ["ROC–AUC", "F1 score", "G-Mean", "Accuracy"]
    grouped = raw.groupby("Resampler")[metrics].agg(["mean", "sem"])
    grouped = grouped.reindex(METHOD_ORDER)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(7.5, 6.8), sharey=True, constrained_layout=True
    )
    y_positions = np.arange(len(METHOD_ORDER))
    colors = ["#8A95A5"] * (len(METHOD_ORDER) - 1) + ["#007C83"]

    for axis, metric, title in zip(axes, metrics, panel_titles):
        means = grouped[(metric, "mean")].to_numpy(dtype=float)
        errors = 1.96 * grouped[(metric, "sem")].fillna(0).to_numpy(dtype=float)
        for row, (mean, error, color) in enumerate(zip(means, errors, colors)):
            axis.errorbar(
                mean,
                row,
                xerr=error,
                fmt="o",
                markersize=4.0 if row < len(METHOD_ORDER) - 1 else 5.2,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
        axis.set_title(title, fontweight="semibold", pad=8)
        axis.set_xlabel("Mean score (95% CI)")
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="x", color="#D9DEE5", linewidth=0.55, alpha=0.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#9AA4B2")
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions, METHOD_TICK_LABELS)
    axes[0].invert_yaxis()
    for label in axes[0].get_yticklabels():
        if label.get_text() == "TopoGeoFlow (Proposed)":
            label.set_color("#007C83")
            label.set_fontweight("bold")

    fig.suptitle(
        "Performance comparison of imbalance-handling methods",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Points show means across cross-validation folds and classifiers; "
        "error bars show 95% confidence intervals.",
        ha="center",
        fontsize=7,
        color="#4B5563",
    )
    png_path = output_dir / "method_comparison_1200dpi.png"
    pdf_path = output_dir / "method_comparison.pdf"
    fig.savefig(
        png_path, dpi=1200, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


# --- ABLATION ---
def save_ablation_figure(raw: pd.DataFrame, output_dir: Path) -> Path:
    """1x4 panel ablation-study figure: mean +/- 95% CI per component set.

    Follows the same aggregation and styling philosophy as
    ``save_comparison_figure`` so the two figure families look consistent in
    a manuscript, but uses the ablation-variant labels/order instead of the
    full method comparison.
    """
    metrics = ["AUC", "F1", "G-Mean", "Accuracy"]
    panel_titles = ["(a) ROC-AUC", "(b) F1 score", "(c) G-Mean", "(d) Accuracy"]
    grouped = raw.groupby("Ablation Variant")[metrics].agg(["mean", "sem"])
    grouped = grouped.reindex(ABLATION_ORDER)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(11.0, 3.4), sharey=True, constrained_layout=True
    )
    y_positions = np.arange(len(ABLATION_ORDER))
    colors = ["#8A95A5"] * (len(ABLATION_ORDER) - 1) + ["#007C83"]

    for axis, metric, title in zip(axes, metrics, panel_titles):
        means = grouped[(metric, "mean")].to_numpy(dtype=float)
        errors = 1.96 * grouped[(metric, "sem")].fillna(0).to_numpy(dtype=float)
        for row, (mean, error, color) in enumerate(zip(means, errors, colors)):
            axis.errorbar(
                mean,
                row,
                xerr=error,
                fmt="o",
                markersize=4.4 if row < len(ABLATION_ORDER) - 1 else 5.8,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
        axis.set_title(title, fontweight="semibold", pad=8)
        axis.set_xlabel("Mean score (95% CI)")
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(axis="x", color="#D9DEE5", linewidth=0.55, alpha=0.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#9AA4B2")
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions, ABLATION_ORDER)
    axes[0].invert_yaxis()
    for label in axes[0].get_yticklabels():
        if label.get_text() == "Full TopoGeoFlow":
            label.set_color("#007C83")
            label.set_fontweight("bold")

    fig.suptitle(
        "Ablation study of TopoGeoFlow components",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.05,
        "Points represent mean performance across cross-validation folds "
        "and classifiers; error bars indicate 95% confidence intervals.",
        ha="center",
        fontsize=7,
        color="#4B5563",
    )
    png_path = output_dir / "topogeoflow_ablation_1200dpi.png"
    pdf_path = output_dir / "topogeoflow_ablation.pdf"
    fig.savefig(
        png_path, dpi=1200, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def save_ablation_delta_figure(raw: pd.DataFrame, output_dir: Path) -> Path:
    """Component-contribution figure: Delta metric = variant - Full TopoGeoFlow.

    Computed directly from the same experimental results as the main
    ablation figure (no fabricated or hard-coded values). A vertical
    reference line marks zero; negative points indicate a performance loss
    from removing the corresponding component(s).
    """
    metrics = ["AUC", "F1", "G-Mean", "Accuracy"]
    panel_titles = ["Delta AUC", "Delta F1", "Delta G-Mean", "Delta Accuracy"]
    grouped = raw.groupby("Ablation Variant")[metrics].mean().reindex(ABLATION_ORDER)
    full_scores = grouped.loc["Full TopoGeoFlow"]
    delta = grouped.subtract(full_scores, axis=1)
    variants = [name for name in ABLATION_ORDER if name != "Full TopoGeoFlow"]
    delta_plot = delta.loc[variants]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(11.0, 3.0), sharey=True, constrained_layout=True
    )
    y_positions = np.arange(len(variants))

    for axis, metric, title in zip(axes, metrics, panel_titles):
        values = delta_plot[metric].to_numpy(dtype=float)
        point_colors = ["#C8375B" if value < 0 else "#2E7D5B" for value in values]
        axis.axvline(0, color="#9AA4B2", linewidth=1.0, zorder=1)
        axis.scatter(
            values,
            y_positions,
            c=point_colors,
            s=32,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        axis.set_title(title, fontweight="semibold", pad=8)
        axis.set_xlabel("Delta vs Full TopoGeoFlow")
        axis.grid(axis="x", color="#D9DEE5", linewidth=0.55, alpha=0.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#9AA4B2")
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions, variants)
    axes[0].invert_yaxis()

    fig.suptitle(
        "Component contribution relative to Full TopoGeoFlow",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.06,
        "Negative values indicate a performance loss after removing the "
        "corresponding component(s); values are means over folds and "
        "classifiers, taken directly from the ablation results.",
        ha="center",
        fontsize=7,
        color="#4B5563",
    )
    png_path = output_dir / "topogeoflow_ablation_delta_1200dpi.png"
    fig.savefig(
        png_path, dpi=1200, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    plt.close(fig)
    return png_path
# --- END ABLATION ---


def save_feature_space_figure(
    X_raw: pd.DataFrame, y: pd.Series, seed: int, output_dir: Path
) -> Path:
    """Plot Original plus all methods as a publication-ready 4x4 PCA grid."""
    preprocessor = make_preprocessor(X_raw)
    X = np.asarray(preprocessor.fit_transform(X_raw), dtype=float)
    y_array = y.to_numpy()
    pca = PCA(n_components=2, random_state=seed)
    original_2d = pca.fit_transform(X)
    labels, counts = np.unique(y_array, return_counts=True)
    minority = labels[np.argmin(counts)]
    majority = labels[np.argmax(counts)]

    panels: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("Original", X, y_array, np.zeros(len(y_array), dtype=bool))
    ]
    for name, sampler in samplers(seed).items():
        X_res, y_res = sampler.fit_resample(X, y_array)
        X_res = np.asarray(X_res, dtype=float)
        y_res = np.asarray(y_res)

        # Continuous synthetic samples differ from every original observation.
        # ROS duplicates are identified from its appended block explicitly.
        if name == "Random Under-Sampling (RUS)":
            synthetic = np.zeros(len(y_res), dtype=bool)
        elif name == "Random Over-Sampling (ROS)":
            synthetic = np.arange(len(y_res)) >= len(y_array)
        else:
            nearest_distance = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(
                X_res, return_distance=True
            )[0][:, 0]
            synthetic = (y_res == minority) & (nearest_distance > 1e-6)
        panels.append((name, X_res, y_res, synthetic))

    fig, axes = plt.subplots(
        4, 4, figsize=(8.5, 8.0), constrained_layout=True
    )
    axes = axes.ravel()
    short_titles = ["Original"] + METHOD_TICK_LABELS
    for index, (axis, panel, short_title) in enumerate(
        zip(axes, panels, short_titles)
    ):
        name, X_res, y_res, synthetic = panel
        projected = original_2d if index == 0 else pca.transform(X_res)
        majority_mask = y_res == majority
        minority_original = (y_res == minority) & ~synthetic

        axis.scatter(
            projected[majority_mask, 0],
            projected[majority_mask, 1],
            s=4.5,
            c="#3B82C4",
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        axis.scatter(
            projected[minority_original, 0],
            projected[minority_original, 1],
            s=12,
            c="#C8375B",
            marker="D",
            alpha=0.82,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
            zorder=3,
        )
        if synthetic.any():
            proposed = name == "TopoGeoFlow (Proposed)"
            axis.scatter(
                projected[synthetic, 0],
                projected[synthetic, 1],
                s=18 if proposed else 10,
                c="#007C83" if proposed else "#F28E2B",
                marker="*" if proposed else "^",
                alpha=0.9 if proposed else 0.58,
                edgecolors="#F2C94C" if proposed else "none",
                linewidths=0.45 if proposed else 0,
                rasterized=True,
                zorder=4,
            )

        title_color = "#007C83" if name == "TopoGeoFlow (Proposed)" else "#172033"
        axis.set_title(
            short_title,
            fontsize=8.2,
            fontweight="bold" if index in (0, len(panels) - 1) else "semibold",
            color=title_color,
            pad=4,
        )
        axis.text(
            0.02,
            0.98,
            f"$n$={len(y_res):,}\nMin={np.sum(y_res == minority):,}\n"
            f"Syn={np.sum(synthetic):,}",
            transform=axis.transAxes,
            va="top",
            fontsize=5.5,
            color="#364152",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#CBD2DC",
                "alpha": 0.82,
                "linewidth": 0.45,
            },
            zorder=6,
        )
        axis.grid(True, color="#D9DEE5", linestyle="--", linewidth=0.35, alpha=0.7)
        axis.set_facecolor("#FAFBFC")
        axis.tick_params(labelsize=5.5, length=2, colors="#5B6573")
        for spine in axis.spines.values():
            spine.set_color("#C7CED8")
            spine.set_linewidth(0.5)
        if index // 4 == 3:
            axis.set_xlabel("PC1", fontsize=6.5)
        if index % 4 == 0:
            axis.set_ylabel("PC2", fontsize=6.5)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markerfacecolor="#3B82C4",
            markeredgecolor="none", markersize=5, label="Majority class"
        ),
        plt.Line2D(
            [0], [0], marker="D", linestyle="", markerfacecolor="#C8375B",
            markeredgecolor="white", markersize=5, label="Original minority"
        ),
        plt.Line2D(
            [0], [0], marker="^", linestyle="", markerfacecolor="#F28E2B",
            markeredgecolor="none", markersize=5, label="Synthetic samples"
        ),
        plt.Line2D(
            [0], [0], marker="*", linestyle="", markerfacecolor="#007C83",
            markeredgecolor="#F2C94C", markersize=7,
            label="TopoGeoFlow synthetic"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(
        "Feature-space comparison of imbalance-handling techniques",
        fontsize=12,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.5,
        0.015,
        "PCA projection fitted on the original preprocessed data",
        ha="center",
        fontsize=6.5,
        color="#5B6573",
    )

    png_path = output_dir / "feature_space_comparison_1200dpi.png"
    pdf_path = output_dir / "feature_space_comparison.pdf"
    fig.savefig(
        png_path,
        dpi=1200,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compress_level": 6},
    )
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def benchmark(
    X: pd.DataFrame, y: pd.Series, folds: int, seed: int
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        preprocessor = make_preprocessor(X_train_raw)
        X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
        X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)

        for sampler_name, sampler in samplers(seed + fold).items():
            start_resampling = time.perf_counter()
            if sampler is None:
                X_res, y_res = X_train, y_train
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
            resampling_seconds = time.perf_counter() - start_resampling

            for classifier_name, estimator in classifiers(seed + fold).items():
                model = clone(estimator)
                start_training = time.perf_counter()
                model.fit(X_res, y_res)
                training_seconds = time.perf_counter() - start_training
                probability = model.predict_proba(X_test)[:, 1]
                prediction = model.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(
                    y_test, prediction, labels=[0, 1]
                ).ravel()
                sensitivity = tp / (tp + fn) if tp + fn else 0.0
                specificity = tn / (tn + fp) if tn + fp else 0.0
                rows.append(
                    {
                        "Dataset": "abalone-17_vs_7-8-9-10",
                        "Fold": fold,
                        "Seed": seed,
                        "Resampler": sampler_name,
                        "Classifier": classifier_name,
                        "AUC": roc_auc_score(y_test, probability),
                        "F1": f1_score(y_test, prediction, zero_division=0),
                        "G-Mean": np.sqrt(sensitivity * specificity),
                        "Accuracy": accuracy_score(y_test, prediction),
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "Train Size Before": len(y_train),
                        "Train Size After": len(y_res),
                        "Resampling Seconds": resampling_seconds,
                        "Training Seconds": training_seconds,
                    }
                )
                print(
                    f"fold={fold} resampler={sampler_name} "
                    f"classifier={classifier_name}",
                    flush=True,
                )
    return pd.DataFrame(rows)


# --- ABLATION ---
def run_ablation_benchmark(
    X: pd.DataFrame, y: pd.Series, folds: int, seed: int
) -> pd.DataFrame:
    """Evaluate every TopoGeoFlow ablation variant under the exact same
    dataset, folds, train/test splits, preprocessing, classifiers, and
    random-seed scheme as ``benchmark``. Oversampling occurs only on the
    training portion of each fold, so no leakage is introduced.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_test = y.iloc[test_idx].to_numpy()
        preprocessor = make_preprocessor(X_train_raw)
        X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
        X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)

        # Same seed-per-fold convention as the main benchmark loop, so
        # ablation variants within a fold differ only by which components
        # are switched on, not by unrelated random variation.
        for variant_name, sampler in ablation_variants(seed + fold).items():
            start_resampling = time.perf_counter()
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            resampling_seconds = time.perf_counter() - start_resampling

            for classifier_name, estimator in classifiers(seed + fold).items():
                model = clone(estimator)
                start_training = time.perf_counter()
                model.fit(X_res, y_res)
                training_seconds = time.perf_counter() - start_training
                probability = model.predict_proba(X_test)[:, 1]
                prediction = model.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(
                    y_test, prediction, labels=[0, 1]
                ).ravel()
                sensitivity = tp / (tp + fn) if tp + fn else 0.0
                specificity = tn / (tn + fp) if tn + fp else 0.0
                rows.append(
                    {
                        "Dataset": "abalone-17_vs_7-8-9-10",
                        "Fold": fold,
                        "Seed": seed,
                        "Ablation Variant": variant_name,
                        "Classifier": classifier_name,
                        "AUC": roc_auc_score(y_test, probability),
                        "F1": f1_score(y_test, prediction, zero_division=0),
                        "G-Mean": np.sqrt(sensitivity * specificity),
                        "Accuracy": accuracy_score(y_test, prediction),
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "Train Size Before": len(y_train),
                        "Train Size After": len(y_res),
                        "Resampling Seconds": resampling_seconds,
                        "Training Seconds": training_seconds,
                    }
                )
                print(
                    f"[ablation] fold={fold} variant={variant_name} "
                    f"classifier={classifier_name}",
                    flush=True,
                )
    return pd.DataFrame(rows)
# --- END ABLATION ---


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # --- ABLATION ---
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the TopoGeoFlow ablation study and only run the main "
        "method-comparison benchmark.",
    )
    # --- END ABLATION ---
    args = parser.parse_args()

    X, y = load_keel_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_figure_path = save_feature_space_figure(
        X, y, args.seed, args.output_dir
    )
    raw = benchmark(X, y, args.folds, args.seed)
    summary = (
        raw.groupby(["Resampler", "Classifier"], as_index=False)
        .agg(
            AUC_Mean=("AUC", "mean"),
            AUC_SD=("AUC", "std"),
            F1_Mean=("F1", "mean"),
            F1_SD=("F1", "std"),
            GMean_Mean=("G-Mean", "mean"),
            GMean_SD=("G-Mean", "std"),
            Accuracy_Mean=("Accuracy", "mean"),
            Accuracy_SD=("Accuracy", "std"),
            Runtime_Seconds=("Training Seconds", "sum"),
        )
        .sort_values(["Classifier", "AUC_Mean"], ascending=[True, False])
    )
    raw_path = save_csv_safely(raw, args.output_dir / "fold_results.csv")
    summary_path = save_csv_safely(
        summary, args.output_dir / "summary_results.csv"
    )
    figure_path = save_comparison_figure(raw, args.output_dir)

    metadata = {
        "dataset": str(args.dataset.resolve()),
        "n_samples": len(y),
        "n_features_raw": X.shape[1],
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "positive_label": "positive",
        "folds": args.folds,
        "seed": args.seed,
        "method_definitions": {
            "RUS": "Randomly removes majority samples until classes are balanced.",
            "ROS": "Randomly duplicates minority samples until classes are balanced.",
            "SMT": "Interpolates synthetic samples between minority k-nearest neighbours.",
            "BS": "Applies SMOTE primarily to minority samples in danger near the boundary.",
            "Ads": "Adaptively generates more samples in difficult minority regions.",
            "SSMT": "Uses an SVM boundary and SMOTE interpolation around support vectors.",
            "SMTN": "Applies SMOTE followed by Edited Nearest Neighbours cleaning.",
            "SMTT": "Applies SMOTE followed by Tomek-link boundary cleaning.",
            "RSMT": "Adaptation using relative-density noise filtering and absolute/sparse density weighting.",
            "TSMT": "Adaptation drawing samples on six directed triangle median lines with variance regulation.",
            "SPYP": "Classification adaptation of the potato-yield study's standard SMOTE augmentation step.",
            "TSP-SMOTE": "Adaptation using scored local minority triangles and locally scaled random perturbation.",
            "SMOTE-Complex": "Adaptation weighting seeds by boundary complexity and local sparsity.",
            "SMOTE-Complex-Clustering": "Cluster-aware variant of the complexity-weighted SMOTE adaptation.",
            "TopoGeoFlow": (
                "Our structural prototype: generates along mutual-minority-kNN "
                "edges, adds locally scaled orthogonal geometry noise, and "
                "rejects candidates that are unsafe near the majority class."
            ),
        },
        "adaptation_warning": (
            "RSMT follows the published high-level SMOTE-RD description. TSMT, "
            "SPYP, TSP-SMOTE, SMOTE-Complex, and SMOTE-Complex-Clustering are "
            "explicitly labelled adaptations because verified reference "
            "implementations were not available. The benchmark "
            "TopoGeoFlow sampler is explicitly a structural prototype, not the "
            "planned neural conditional-flow implementation."
        ),
        "fold_results_file": raw_path.name,
        "summary_results_file": summary_path.name,
        "comparison_figure_file": figure_path.name,
        "feature_space_figure_file": feature_figure_path.name,
    }

    generated_files = [
        raw_path,
        summary_path,
        figure_path,
        feature_figure_path,
    ]

    # --- ABLATION ---
    if not args.skip_ablation:
        ablation_raw = run_ablation_benchmark(X, y, args.folds, args.seed)
        ablation_summary = (
            ablation_raw.groupby(
                ["Ablation Variant", "Classifier"], as_index=False
            )
            .agg(
                AUC_Mean=("AUC", "mean"),
                AUC_SD=("AUC", "std"),
                F1_Mean=("F1", "mean"),
                F1_SD=("F1", "std"),
                GMean_Mean=("G-Mean", "mean"),
                GMean_SD=("G-Mean", "std"),
                Accuracy_Mean=("Accuracy", "mean"),
                Accuracy_SD=("Accuracy", "std"),
                Runtime_Seconds=("Training Seconds", "sum"),
            )
            .sort_values(["Classifier", "AUC_Mean"], ascending=[True, False])
        )
        ablation_raw_path = save_csv_safely(
            ablation_raw, args.output_dir / "ablation_fold_results.csv"
        )
        ablation_summary_path = save_csv_safely(
            ablation_summary, args.output_dir / "ablation_summary_results.csv"
        )
        ablation_figure_path = save_ablation_figure(ablation_raw, args.output_dir)
        ablation_delta_figure_path = save_ablation_delta_figure(
            ablation_raw, args.output_dir
        )

        metadata["ablation_variant_definitions"] = {
            "Basic interpolation": (
                "All three proposed components disabled: ordinary "
                "minority-neighbour interpolation only."
            ),
            "+ Topology": (
                "Mutual-kNN neighbour selection enabled; geometry noise and "
                "boundary rejection disabled."
            ),
            "+ Topology + Geometry": (
                "Mutual-kNN neighbour selection and locally scaled orthogonal "
                "geometry noise enabled; boundary rejection disabled."
            ),
            "+ Topology + Boundary": (
                "Mutual-kNN neighbour selection and boundary-safety rejection "
                "enabled; geometry noise disabled."
            ),
            "+ Geometry + Boundary": (
                "Locally scaled orthogonal geometry noise and boundary-safety "
                "rejection enabled; ordinary (non-mutual) kNN neighbours used "
                "in place of the topology component."
            ),
            "Full TopoGeoFlow": (
                "All three proposed components enabled: mutual-kNN topology, "
                "locally scaled orthogonal geometry noise, and majority-"
                "boundary safety rejection. Mathematically identical to the "
                "non-configurable TopoGeoFlow implementation."
            ),
        }
        metadata["ablation_fold_results_file"] = ablation_raw_path.name
        metadata["ablation_summary_results_file"] = ablation_summary_path.name
        metadata["ablation_figure_file"] = ablation_figure_path.name
        metadata["ablation_delta_figure_file"] = ablation_delta_figure_path.name

        generated_files.extend(
            [
                ablation_raw_path,
                ablation_summary_path,
                ablation_figure_path,
                ablation_delta_figure_path,
            ]
        )
    # --- END ABLATION ---

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved benchmark outputs to {args.output_dir}")
    for path in generated_files:
        print(f"Generated file: {path}")


if __name__ == "__main__":
    main()