"""Small in-memory flat-file ANN over f32 vectors."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from importlib import metadata
from typing import Final

import numpy as np
from numpy.typing import NDArray

from annflat._native import AnnflatError, Hit
from annflat._native import Index as _NativeIndex


def _read_version() -> str:
    try:
        return metadata.version("annflat")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__: Final[str] = _read_version()

__all__ = ["AnnflatError", "Hit", "Index", "Metric", "__version__"]


class Metric(str, Enum):
    """Distance metric. String-valued so it's JSON-friendly."""

    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"


def _as_f32_1d(name: str, arr: NDArray[np.float32]) -> NDArray[np.float32]:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def _as_f32_2d(name: str, arr: NDArray[np.float32]) -> NDArray[np.float32]:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


class Index:
    """Flat-file ANN index."""

    def __init__(self, *, metric: Metric | str = Metric.COSINE) -> None:
        m = metric.value if isinstance(metric, Metric) else str(metric)
        self._inner = _NativeIndex(metric=m)

    @property
    def metric(self) -> str:
        return str(self._inner.metric)

    @property
    def dim(self) -> int | None:
        d = self._inner.dim
        return None if d is None else int(d)

    def add(self, id: str, vector: NDArray[np.float32]) -> None:
        """Insert a single vector."""
        self._inner.add(id, _as_f32_1d("vector", vector))

    def add_batch(self, ids: Sequence[str], matrix: NDArray[np.float32]) -> None:
        """Insert many vectors. `matrix` is `(n, d)`."""
        self._inner.add_batch(list(ids), _as_f32_2d("matrix", matrix))

    def search(self, query: NDArray[np.float32], k: int) -> list[Hit]:
        """Top-k search."""
        return list(self._inner.search(_as_f32_1d("query", query), k))

    def search_batch(
        self,
        queries: NDArray[np.float32],
        k: int,
        *,
        parallel: bool = False,
    ) -> list[list[Hit]]:
        """Top-k for many queries. `queries` is `(n_queries, d)`."""
        raw = self._inner.search_batch(_as_f32_2d("queries", queries), k, parallel=parallel)
        return [list(row) for row in raw]

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        return repr(self._inner)
