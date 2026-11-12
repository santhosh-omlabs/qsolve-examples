"""Turn the workflow input into a train and test split with features scaled for the feature map."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qsolve import WorkflowError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

MAX_TRAIN = 60  # kernel circuits grow with its square
REAL = {"breast_cancer"}  # real measurements, bundled with scikit-learn (no download)


@dataclass(frozen=True)
class Split:
    X_train: FloatArray
    y_train: IntArray
    X_test: FloatArray
    y_test: IntArray


def _real(n: int, seed: int) -> tuple[FloatArray, IntArray]:
    """A stratified random sample of n rows from the UCI Wisconsin Diagnostic Breast Cancer data
    (CC BY 4.0), all of its features."""
    data = load_breast_cancer()
    X, y = np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=int)
    X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X, y


def _reduce(Xtr: FloatArray, Xte: FloatArray, k: int) -> tuple[FloatArray, FloatArray]:
    """Standardise and project onto k principal components, fitted on the training rows only so
    nothing about the test set leaks into the features."""
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=k, random_state=0).fit(scaler.transform(Xtr))
    return pca.transform(scaler.transform(Xtr)), pca.transform(scaler.transform(Xte))


def _generate(spec: dict[str, Any]) -> tuple[FloatArray, IntArray]:
    kind, n = spec["kind"], int(spec.get("n_samples", 36))
    noise, seed = float(spec.get("noise", 0.15)), int(spec.get("seed", 0))
    if kind in REAL:
        return _real(n, seed)
    if kind == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif kind == "circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.4, random_state=seed)
    elif kind == "blobs":
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=1.0 + 2 * noise, random_state=seed)
    else:  # xor
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X = X + rng.normal(0, noise * 0.3, X.shape)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _table(t: dict[str, Any]) -> tuple[FloatArray, IntArray]:
    X, y = np.asarray(t["X"], dtype=float), np.asarray(t["y"], dtype=int)
    if X.ndim != 2 or len(X) != len(y):
        raise WorkflowError("X must be a rectangular table with one label per row")
    return X, y


def scale_features(
    X_train: FloatArray, X_test: FloatArray, lo: float = 0.0, hi: float = math.pi
) -> tuple[FloatArray, FloatArray]:
    """Min-max scale to [lo, hi] using training statistics only (no test leakage)."""
    mn, mx = X_train.min(axis=0), X_train.max(axis=0)
    span = np.where(mx - mn < 1e-12, 1.0, mx - mn)

    def f(X: FloatArray) -> FloatArray:
        return np.asarray(lo + (hi - lo) * np.clip((X - mn) / span, 0.0, 1.0), dtype=np.float64)

    return f(X_train), f(X_test)


def build_split(input: dict[str, Any]) -> Split:
    """The train and test sets for an input: a sample dataset, one table split for you, or your
    own train and test tables. Raises `WorkflowError` for anything the user got wrong."""
    frac = float(input.get("test_fraction", 0.33))  # the schema's default
    if "train" in input:
        if "test" not in input:
            raise WorkflowError("provide both 'train' and 'test', or a single 'data' table")
        Xtr, ytr = _table(input["train"])
        Xte, yte = _table(input["test"])
    else:
        X, y = _generate(input["dataset"]) if "dataset" in input else _table(input["data"])
        if len(np.unique(y)) < 2:
            raise WorkflowError("data must contain both classes")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=frac, stratify=y, random_state=0)
    if len(Xtr) > MAX_TRAIN:
        raise WorkflowError(f"at most {MAX_TRAIN} training points per run, got {len(Xtr)}")
    spec = input.get("dataset") or {}
    if spec.get("kind") in REAL:
        Xtr, Xte = _reduce(Xtr, Xte, int(spec.get("features", 4)))
    Xtr, Xte = scale_features(Xtr, Xte)
    if len(np.unique(ytr)) < 2:
        raise WorkflowError("training data must contain both classes")
    if (set(np.unique(ytr)) | set(np.unique(yte))) - {0, 1}:
        raise WorkflowError("labels must be 0 or 1")
    if Xtr.shape[1] != Xte.shape[1]:
        raise WorkflowError("train and test must have the same number of features")
    return Split(Xtr, ytr, Xte, yte)
