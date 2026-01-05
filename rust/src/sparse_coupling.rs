use pyo3::{
    types::{PyAnyMethods, PyDict, PyDictMethods, PyInt, PyTuple},
    Bound, IntoPyObject, PyErr, PyResult, Python,
};
use std::collections::BinaryHeap;

// Tolerance for considering values as zero in coupling construction
const COUPLING_TOLERANCE: f64 = 1e-12;
// Tolerance for residual warning threshold
const RESIDUAL_WARNING_THRESHOLD: f64 = 1e-6;

/// A sparse representation of a joint probability distribution (coupling).
///
/// Stores only non-zero entries as (indices, value) pairs where indices
/// represent the coordinates in the multi-dimensional distribution.
#[derive(Debug, Clone)]
pub struct SparseCoupling(pub Vec<(Vec<usize>, f64)>);

impl<'py> IntoPyObject<'py> for SparseCoupling {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let out = PyDict::new(py);
        for (indices, value) in self.0 {
            let py_indices = indices.into_iter().map(|idx| PyInt::new(py, idx));
            let indices_tuple = PyTuple::new(py, py_indices)?;
            out.set_item(indices_tuple, value).unwrap();
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct ValAndIdx(f64, usize);

impl Eq for ValAndIdx {}
impl Ord for ValAndIdx {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("NaN values are not allowed")
    }
}

impl SparseCoupling {
    pub fn new(py: Python, ndarrays: &[numpy::PyReadonlyArray1<f64>]) -> PyResult<Self> {
        if ndarrays.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected at least one marginal distribution"
            ));
        }
        let mut data = vec![];

        let mut heaps: Vec<BinaryHeap<ValAndIdx>> = ndarrays
            .iter()
            .map(|arr| {
                let view = arr.as_array();

                let mut heap = BinaryHeap::new();
                for (idx, &val) in view.iter().enumerate() {
                    if val > COUPLING_TOLERANCE {
                        heap.push(ValAndIdx(val, idx));
                    }
                }
                heap
            })
            .collect();

        while heaps.iter().all(|heap| !heap.is_empty()) {
            let tops: Vec<ValAndIdx> = heaps
                .iter_mut()
                .map(|heap| heap.pop().expect("The heap is not empty"))
                .collect();
            let min_val = tops.iter().min().expect("There is at least one top").0;

            heaps.iter_mut().zip(tops.iter()).for_each(|(heap, top)| {
                let remaining = top.0 - min_val;
                if remaining > COUPLING_TOLERANCE {
                    heap.push(ValAndIdx(remaining, top.1));
                }
            });

            data.push((tops.into_iter().map(|val_idx| val_idx.1).collect(), min_val));
        }

        let residual = heaps
            .into_iter()
            .map(|heap| heap.into_iter().map(|val_idx| val_idx.0).sum::<f64>())
            .sum::<f64>();
        if residual > RESIDUAL_WARNING_THRESHOLD {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "Residual value after sparse coupling construction is: {}",
                    residual
                ),),
            )?;
        }

        Ok(SparseCoupling(data))
    }
}
