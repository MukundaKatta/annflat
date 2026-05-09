"""End-to-end tests for annflat."""

from __future__ import annotations

import numpy as np
import pytest
from annflat import AnnflatError, Hit, Index, Metric, __version__


def test_version_present() -> None:
    assert isinstance(__version__, str) and __version__ != ""


def test_empty_index_state() -> None:
    idx = Index()
    assert len(idx) == 0
    assert idx.dim is None
    assert idx.metric == "cosine"


def test_cosine_search_finds_self() -> None:
    idx = Index(metric=Metric.COSINE)
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([0.0, 1.0], dtype=np.float32))
    idx.add("c", np.array([0.7071, 0.7071], dtype=np.float32))
    hits = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert hits[0].id == "a"
    assert abs(hits[0].score - 1.0) < 1e-4


def test_l2_smaller_distance_first() -> None:
    idx = Index(metric=Metric.L2)
    idx.add("near", np.array([1.0, 1.0], dtype=np.float32))
    idx.add("far", np.array([10.0, 10.0], dtype=np.float32))
    hits = idx.search(np.array([1.0, 1.1], dtype=np.float32), k=2)
    assert hits[0].id == "near"
    assert hits[0].score > hits[1].score


def test_dot_product_search() -> None:
    idx = Index(metric=Metric.DOT)
    idx.add("a", np.array([1.0, 1.0], dtype=np.float32))
    idx.add("b", np.array([2.0, 2.0], dtype=np.float32))
    hits = idx.search(np.array([1.0, 1.0], dtype=np.float32), k=2)
    assert hits[0].id == "b"


def test_dim_set_on_first_insert() -> None:
    idx = Index()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    assert idx.dim == 2


def test_dim_mismatch_rejected() -> None:
    idx = Index()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        idx.add("b", np.array([1.0], dtype=np.float32))


def test_add_batch() -> None:
    idx = Index()
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    idx.add_batch(["a", "b", "c"], matrix)
    assert len(idx) == 3


def test_add_batch_length_mismatch() -> None:
    idx = Index()
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        idx.add_batch(["only_one"], matrix)


def test_unknown_metric_rejected() -> None:
    with pytest.raises(ValueError):
        Index(metric="manhattan")


def test_k_zero_rejected() -> None:
    idx = Index()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        idx.search(np.array([1.0, 0.0], dtype=np.float32), k=0)


def test_k_too_large_rejected() -> None:
    idx = Index()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        idx.search(np.array([1.0, 0.0], dtype=np.float32), k=5)


def test_search_batch_serial_and_parallel_match() -> None:
    rng = np.random.default_rng(0)
    idx = Index(metric=Metric.COSINE)
    matrix = rng.standard_normal((50, 4)).astype(np.float32)
    ids = [f"d{i}" for i in range(50)]
    idx.add_batch(ids, matrix)
    queries = rng.standard_normal((5, 4)).astype(np.float32)
    s = idx.search_batch(queries, k=3)
    p = idx.search_batch(queries, k=3, parallel=True)
    assert len(s) == 5 and len(p) == 5
    for sr, pr in zip(s, p, strict=True):
        assert [(h.id, round(h.score, 4)) for h in sr] == [(h.id, round(h.score, 4)) for h in pr]


def test_dtype_coerced_to_float32() -> None:
    idx = Index()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float64))
    hits = idx.search(np.array([1.0, 0.0], dtype=np.float64), k=1)
    assert hits[0].id == "a"


def test_repr_includes_metric_and_n() -> None:
    idx = Index(metric=Metric.L2)
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    text = repr(idx)
    assert "l2" in text
    assert "n=1" in text


def test_native_error_class_exposed() -> None:
    assert issubclass(AnnflatError, Exception)


def test_hit_dataclass() -> None:
    idx = Index()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    h = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=1)[0]
    assert isinstance(h, Hit)
    assert h.id == "a"


def test_cosine_normalizes_at_insert() -> None:
    idx = Index(metric=Metric.COSINE)
    idx.add("a", np.array([3.0, 4.0], dtype=np.float32))  # norm 5
    hits = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=1)
    assert abs(hits[0].score - 0.6) < 1e-4
