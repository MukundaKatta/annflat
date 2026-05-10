//! PyO3 bindings exposing `annflat_core` as `annflat._native`.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use annflat_core::{AnnFlatError, Hit, Index, Metric};

pyo3::create_exception!(_native, AnnFlatError_Py, pyo3::exceptions::PyException);

fn map_err(e: AnnFlatError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn metric_from_str(s: &str) -> PyResult<Metric> {
    Ok(match s {
        "cosine" => Metric::Cosine,
        "l2" => Metric::L2,
        "dot" => Metric::Dot,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown metric: {other:?} (expected 'cosine', 'l2', or 'dot')"
            )));
        }
    })
}

fn metric_to_str(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::L2 => "l2",
        Metric::Dot => "dot",
    }
}

#[pyclass(name = "Hit", module = "annflat._native", frozen)]
#[derive(Clone)]
struct PyHit {
    inner: Hit,
}

#[pymethods]
impl PyHit {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }
    fn __repr__(&self) -> String {
        format!("Hit(id={:?}, score={:.4})", self.inner.id, self.inner.score)
    }
}

#[pyclass(name = "Index", module = "annflat._native")]
struct PyIndex {
    inner: Index,
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (*, metric="cosine"))]
    fn new(metric: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Index::new(metric_from_str(metric)?),
        })
    }

    #[getter]
    fn metric(&self) -> &'static str {
        metric_to_str(self.inner.metric())
    }

    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim()
    }

    fn add(&mut self, py: Python<'_>, id: &str, vector: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let id = id.to_owned();
        let v: Vec<f32> = vector.as_slice()?.to_vec();
        py.allow_threads(move || self.inner.add(id, &v))
            .map_err(map_err)
    }

    fn add_batch(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        matrix: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<()> {
        let owned = matrix.as_array().to_owned();
        py.allow_threads(move || self.inner.add_batch(ids, &owned.view()))
            .map_err(map_err)
    }

    fn search(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<'_, f32>,
        k: usize,
    ) -> PyResult<Vec<PyHit>> {
        let q: Vec<f32> = query.as_slice()?.to_vec();
        let raw = py
            .allow_threads(move || self.inner.search(&q, k))
            .map_err(map_err)?;
        Ok(raw.into_iter().map(|inner| PyHit { inner }).collect())
    }

    #[pyo3(signature = (queries, k, *, parallel=false))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray2<'_, f32>,
        k: usize,
        parallel: bool,
    ) -> PyResult<Vec<Vec<PyHit>>> {
        let owned = queries.as_array().to_owned();
        let raw = py
            .allow_threads(move || self.inner.search_batch(&owned.view(), k, parallel))
            .map_err(map_err)?;
        Ok(raw
            .into_iter()
            .map(|v| v.into_iter().map(|inner| PyHit { inner }).collect())
            .collect())
    }

    fn remove(&mut self, py: Python<'_>, id: &str) -> bool {
        let owned = id.to_owned();
        py.allow_threads(move || self.inner.remove(&owned))
    }

    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        py.allow_threads(move || self.inner.save(path))
            .map_err(map_err)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let inner = py
            .allow_threads(move || annflat_core::Index::load(path))
            .map_err(map_err)?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        Ok(PyString::new(
            py,
            &format!(
                "Index(metric='{}', n={})",
                metric_to_str(self.inner.metric()),
                self.inner.len()
            ),
        ))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("AnnflatError", m.py().get_type::<AnnFlatError_Py>())?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyIndex>()?;
    Ok(())
}
