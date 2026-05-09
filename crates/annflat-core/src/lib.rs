//! Pure-Rust core for `annflat`. Flat brute-force ANN over `(n, d)` f32
//! vectors with cosine / L2 / inner-product metrics.
//!
//! Cosine is implemented as inner product over L2-normalized vectors, so
//! adding under the cosine metric runs an in-place normalization once at
//! insert time. Subsequent search is a single matrix-vector dot.
//!
//! `score` is "higher is better": for cosine it's the cosine similarity
//! in `[-1, 1]`, for inner product it's the dot, for L2 it's `-distance`
//! (so the same `top_k` heap logic works across metrics).

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, AnnFlatError>;

/// All errors surfaced by `annflat-core`.
#[derive(Error, Debug)]
pub enum AnnFlatError {
    /// Caller supplied a vector of the wrong dimensionality.
    #[error("dim mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Expected dimensionality (set by the first insert).
        expected: usize,
        /// Provided dimensionality.
        got: usize,
    },
    /// Caller asked for more results than exist in the index.
    #[error("k ({k}) > index size ({n})")]
    KTooLarge {
        /// Requested k.
        k: usize,
        /// Available entries.
        n: usize,
    },
    /// Caller passed `k = 0`.
    #[error("k must be > 0")]
    KZero,
    /// `add_batch` length mismatch.
    #[error("add_batch ids and matrix row counts disagree: {ids} vs {rows}")]
    BatchLengthMismatch {
        /// Number of ids supplied.
        ids: usize,
        /// Number of rows in the matrix.
        rows: usize,
    },
}

/// Distance metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Cosine similarity. Vectors are L2-normalized at insert time.
    Cosine,
    /// L2 (Euclidean) distance. Reported `score` is `-distance`.
    L2,
    /// Inner product. No normalization.
    Dot,
}

/// One match returned by [`Index::search`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    /// Document id.
    pub id: String,
    /// Score; higher is better. See [`Metric`] for semantics.
    pub score: f32,
}

/// In-memory flat ANN index.
pub struct Index {
    metric: Metric,
    dim: Option<usize>,
    ids: Vec<String>,
    /// `(n, d)` f32 storage. For `Metric::Cosine`, rows are L2-normalized.
    vectors: Vec<Vec<f32>>,
}

impl Index {
    /// Build an empty index.
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            dim: None,
            ids: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Active metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True iff no vectors are indexed.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Dimension of stored vectors, set on first insert.
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Insert a single vector.
    pub fn add(&mut self, id: impl Into<String>, vector: &[f32]) -> Result<()> {
        match self.dim {
            None => self.dim = Some(vector.len()),
            Some(d) if d != vector.len() => {
                return Err(AnnFlatError::DimMismatch {
                    expected: d,
                    got: vector.len(),
                });
            }
            _ => {}
        }
        let mut v = vector.to_vec();
        if self.metric == Metric::Cosine {
            normalize_in_place(&mut v);
        }
        self.ids.push(id.into());
        self.vectors.push(v);
        Ok(())
    }

    /// Insert many vectors. `matrix` is `(n, d)`.
    pub fn add_batch(&mut self, ids: Vec<String>, matrix: &ArrayView2<'_, f32>) -> Result<()> {
        if ids.len() != matrix.nrows() {
            return Err(AnnFlatError::BatchLengthMismatch {
                ids: ids.len(),
                rows: matrix.nrows(),
            });
        }
        for (id, row) in ids.into_iter().zip(matrix.axis_iter(Axis(0))) {
            self.add(id, row.as_slice().unwrap_or(&row.to_vec()))?;
        }
        Ok(())
    }

    /// Top-k search. `query` is a 1-D vector of the same dimension as
    /// stored vectors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        if k == 0 {
            return Err(AnnFlatError::KZero);
        }
        if k > self.len() {
            return Err(AnnFlatError::KTooLarge { k, n: self.len() });
        }
        match self.dim {
            Some(d) if d != query.len() => {
                return Err(AnnFlatError::DimMismatch {
                    expected: d,
                    got: query.len(),
                });
            }
            None => {
                return Err(AnnFlatError::KTooLarge { k, n: 0 });
            }
            _ => {}
        }
        let q: Vec<f32> = if self.metric == Metric::Cosine {
            let mut q2 = query.to_vec();
            normalize_in_place(&mut q2);
            q2
        } else {
            query.to_vec()
        };

        // Maintain a min-heap of (score, idx) of size k. Use OrdScore so we
        // can BinaryHeap on f32.
        let mut heap: BinaryHeap<(Reverse<OrdScore>, usize)> = BinaryHeap::with_capacity(k);
        for (i, v) in self.vectors.iter().enumerate() {
            let s = self.score(&q, v);
            let entry = (Reverse(OrdScore(s)), i);
            if heap.len() < k {
                heap.push(entry);
            } else if let Some(top) = heap.peek() {
                if entry.0 < top.0 {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }
        let mut out: Vec<Hit> = heap
            .into_iter()
            .map(|(rs, i)| Hit {
                id: self.ids[i].clone(),
                score: rs.0 .0,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        Ok(out)
    }

    /// Batch search. `queries` is `(n_q, d)`. With `parallel = true`, each
    /// query runs on a rayon thread.
    pub fn search_batch(
        &self,
        queries: &ArrayView2<'_, f32>,
        k: usize,
        parallel: bool,
    ) -> Result<Vec<Vec<Hit>>> {
        if parallel {
            queries
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| self.search_view(&row, k))
                .collect()
        } else {
            queries
                .axis_iter(Axis(0))
                .map(|row| self.search_view(&row, k))
                .collect()
        }
    }

    fn search_view(&self, row: &ArrayView1<'_, f32>, k: usize) -> Result<Vec<Hit>> {
        match row.as_slice() {
            Some(s) => self.search(s, k),
            None => self.search(&row.to_vec(), k),
        }
    }

    /// Snapshot vectors as a `(n, d)` matrix (allocates a copy).
    pub fn vectors(&self) -> Result<Array2<f32>> {
        let n = self.len();
        let d = self.dim.unwrap_or(0);
        if n == 0 {
            return Ok(Array2::<f32>::zeros((0, 0)));
        }
        let mut out = Array2::<f32>::zeros((n, d));
        for (i, v) in self.vectors.iter().enumerate() {
            for (j, &x) in v.iter().enumerate() {
                out[[i, j]] = x;
            }
        }
        Ok(out)
    }

    fn score(&self, q: &[f32], v: &[f32]) -> f32 {
        match self.metric {
            Metric::Cosine | Metric::Dot => {
                let mut s = 0.0_f32;
                for (a, b) in q.iter().zip(v.iter()) {
                    s += a * b;
                }
                s
            }
            Metric::L2 => {
                let mut s = 0.0_f32;
                for (a, b) in q.iter().zip(v.iter()) {
                    let d = a - b;
                    s += d * d;
                }
                -s.sqrt()
            }
        }
    }
}

fn normalize_in_place(v: &mut [f32]) {
    let mut sq = 0.0_f32;
    for &x in v.iter() {
        sq += x * x;
    }
    let n = sq.sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    } else {
        for x in v.iter_mut() {
            *x = 0.0;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdScore(f32);
impl Eq for OrdScore {}
impl Ord for OrdScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for OrdScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn empty_search_rejected() {
        let idx = Index::new(Metric::Cosine);
        assert!(idx.search(&[1.0, 2.0], 1).is_err());
    }

    #[test]
    fn cosine_search_finds_self() {
        let mut idx = Index::new(Metric::Cosine);
        idx.add("a", &[1.0, 0.0]).unwrap();
        idx.add("b", &[0.0, 1.0]).unwrap();
        idx.add("c", &[0.6, 0.8]).unwrap();
        let hits = idx.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(hits[0].id, "a");
        assert!((hits[0].score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn l2_search_smaller_distance_first() {
        let mut idx = Index::new(Metric::L2);
        idx.add("near", &[1.0, 1.0]).unwrap();
        idx.add("far", &[10.0, 10.0]).unwrap();
        let hits = idx.search(&[1.0, 1.1], 2).unwrap();
        assert_eq!(hits[0].id, "near");
        // Score is -distance, so near is closer to 0.
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn dot_search() {
        let mut idx = Index::new(Metric::Dot);
        idx.add("a", &[1.0, 1.0]).unwrap();
        idx.add("b", &[2.0, 2.0]).unwrap();
        let hits = idx.search(&[1.0, 1.0], 2).unwrap();
        // b · q = 4, a · q = 2.
        assert_eq!(hits[0].id, "b");
        assert!((hits[0].score - 4.0).abs() < 1e-6);
    }

    #[test]
    fn dim_mismatch_on_add() {
        let mut idx = Index::new(Metric::Cosine);
        idx.add("a", &[1.0, 0.0]).unwrap();
        assert!(idx.add("b", &[1.0]).is_err());
    }

    #[test]
    fn dim_mismatch_on_search() {
        let mut idx = Index::new(Metric::Cosine);
        idx.add("a", &[1.0, 0.0]).unwrap();
        assert!(idx.search(&[1.0], 1).is_err());
    }

    #[test]
    fn k_zero_rejected() {
        let mut idx = Index::new(Metric::Cosine);
        idx.add("a", &[1.0, 0.0]).unwrap();
        assert!(matches!(
            idx.search(&[1.0, 0.0], 0),
            Err(AnnFlatError::KZero)
        ));
    }

    #[test]
    fn k_too_large_rejected() {
        let mut idx = Index::new(Metric::Cosine);
        idx.add("a", &[1.0, 0.0]).unwrap();
        assert!(matches!(
            idx.search(&[1.0, 0.0], 5),
            Err(AnnFlatError::KTooLarge { .. })
        ));
    }

    #[test]
    fn add_batch_works() {
        let mut idx = Index::new(Metric::Cosine);
        let m = arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [0.5, 0.5]]);
        idx.add_batch(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            &m.view(),
        )
        .unwrap();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn add_batch_length_mismatch() {
        let mut idx = Index::new(Metric::Cosine);
        let m = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
        let r = idx.add_batch(vec!["a".to_string()], &m.view());
        assert!(matches!(r, Err(AnnFlatError::BatchLengthMismatch { .. })));
    }

    #[test]
    fn search_batch_serial_and_parallel_match() {
        let mut idx = Index::new(Metric::Cosine);
        for i in 0..50 {
            idx.add(format!("d{i}"), &[i as f32, 1.0, 2.0]).unwrap();
        }
        let q = arr2(&[[1.0_f32, 1.0, 2.0], [25.0, 1.0, 2.0]]);
        let s = idx.search_batch(&q.view(), 5, false).unwrap();
        let p = idx.search_batch(&q.view(), 5, true).unwrap();
        assert_eq!(s, p);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].len(), 5);
    }

    #[test]
    fn metric_get() {
        let idx = Index::new(Metric::L2);
        assert_eq!(idx.metric(), Metric::L2);
    }

    #[test]
    fn empty_index_dim_is_none() {
        let idx = Index::new(Metric::Cosine);
        assert!(idx.dim().is_none());
        assert!(idx.is_empty());
    }

    #[test]
    fn cosine_normalizes_at_insert() {
        let mut idx = Index::new(Metric::Cosine);
        // Insert a non-unit vector. Internally it gets normalized.
        idx.add("a", &[3.0, 4.0]).unwrap();
        // Search with [1, 0]; cosine is 3/5 = 0.6.
        let hits = idx.search(&[1.0, 0.0], 1).unwrap();
        assert!((hits[0].score - 0.6).abs() < 1e-4);
    }
}
