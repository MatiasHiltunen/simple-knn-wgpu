#![cfg(feature = "python")]
use crate::{GpuContext, compute_knn};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn map_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

#[pyclass]
struct PyGpuContext {
    ctx: GpuContext,
}

#[pymethods]
impl PyGpuContext {
    #[new]
    fn new() -> PyResult<Self> {
        let ctx = pollster::block_on(GpuContext::new()).map_err(map_err)?;
        Ok(Self { ctx })
    }

    fn device_description(&self) -> String {
        self.ctx.device_description()
    }

    fn compute_knn<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let points_vec = points.as_slice()?.to_vec();
        let result = pollster::block_on(compute_knn(&self.ctx, &points_vec)).map_err(map_err)?;
        Ok(PyArray1::from_vec(py, result.distances))
    }
}

#[pyfunction]
fn compute_knn_blocking<'py>(
    py: Python<'py>,
    points: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let ctx = pollster::block_on(GpuContext::new()).map_err(map_err)?;
    let points_vec = points.as_slice()?.to_vec();
    let result = pollster::block_on(compute_knn(&ctx, &points_vec)).map_err(map_err)?;
    Ok(PyArray1::from_vec(py, result.distances))
}

#[pymodule]
fn simple_knn_wgpu<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuContext>()?;
    m.add_function(wrap_pyfunction!(compute_knn_blocking, m)?)?;
    Ok(())
}
