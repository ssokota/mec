use pyo3::{
    types::{PyDict, PyDictMethods, PyInt, PyTuple},
    Bound, IntoPyObject, PyErr, PyResult, Python,
};
use std::collections::BinaryHeap;

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
    pub fn new(ndarrays: &[numpy::PyReadonlyArray1<f64>]) -> Self {
        let mut data = vec![];

        let mut heaps: Vec<BinaryHeap<ValAndIdx>> = ndarrays
            .iter()
            .map(|arr| {
                let view = arr.as_array();

                let mut heap = BinaryHeap::new();
                for (idx, &val) in view.iter().enumerate() {
                    if val > 1e-12 {
                        heap.push(ValAndIdx(val, idx));
                    }
                }
                heap
            })
            .collect();

        loop {
            let any_empty = heaps.iter().any(|heap| heap.is_empty());

            if !any_empty {
                let tops: Vec<ValAndIdx> = heaps
                    .iter_mut()
                    .map(|heap| heap.pop().expect("The heap is not empty"))
                    .collect();
                let min_val = tops.iter().min().expect("There is at least one top").0;

                heaps.iter_mut().zip(tops.iter()).for_each(|(heap, top)| {
                    let remaining = top.0 - min_val;
                    if remaining > 1e-12 {
                        heap.push(ValAndIdx(remaining, top.1));
                    }
                });

                data.push((tops.into_iter().map(|val_idx| val_idx.1).collect(), min_val));
            } else {
                break;
            }
        }

        let residual = heaps
            .into_iter()
            .map(|heap| heap.into_iter().map(|val_idx| val_idx.0).sum::<f64>())
            .sum::<f64>();
        if residual > 1e-6 {
            println!(
                "[WARN] residual value after sparse coupling construction is: {}",
                residual
            );
        }

        SparseCoupling(data)
    }
}
