use numpy::{
    convert::IntoPyArray,
    ndarray::{Array, Dim, IxDynImpl},
    PyArray1, PyArrayMethods,
};
use pyo3::{prelude::*, types::PyTuple};
mod sparse_coupling;
use sparse_coupling::SparseCoupling;

#[derive(Debug)]
enum SparseCouplingOrMatrix {
    Sparse(SparseCoupling),
    Dense(Array<f64, Dim<IxDynImpl>>),
}

impl<'py> IntoPyObject<'py> for SparseCouplingOrMatrix {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            SparseCouplingOrMatrix::Sparse(sparse) => {
                sparse.into_pyobject(py).map(|obj| obj.into_any())
            }
            SparseCouplingOrMatrix::Dense(matrix) => Ok(matrix.into_pyarray(py).into_any()),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (*marginals, sparse=false))]
fn greedy_mec(marginals: &Bound<'_, PyTuple>, sparse: bool) -> PyResult<SparseCouplingOrMatrix> {
    if marginals.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Expected at least two marginals",
        ));
    }
    let mut ndarrays = vec![];
    for marginal in marginals {
        match marginal.cast::<PyArray1<f64>>() {
            Ok(obj) => ndarrays.push(obj.readonly()),
            Err(_) => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Each marginal must be a 1-dimensional numpy array of type float64",
                ));
            }
        }
    }

    // Check that all the arrays are valid probability distributions
    for (idx, arr) in ndarrays.iter().enumerate() {
        let view = arr.as_array();
        let sum: f64 = view.sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Marginal #{idx} sums to {sum} != 1.0"
            )));
        }
        for (j, &val) in view.iter().enumerate() {
            if val < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Marginal #{idx} contains negative value {val} at index {j}",
                )));
            }
            if !val.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Marginal #{idx} contains non-finite value {val} at index {j}",
                )));
            }
        }
    }

    let gamma = SparseCoupling::new(&ndarrays);

    if sparse {
        Ok(SparseCouplingOrMatrix::Sparse(gamma))
    } else {
        let shapes = ndarrays
            .iter()
            .map(|arr| arr.as_array().len())
            .collect::<Vec<usize>>();
        let strides = {
            let mut strides = vec![1usize; shapes.len()];
            for i in (0..shapes.len() - 1).rev() {
                strides[i] = strides[i + 1] * shapes[i + 1];
            }
            strides
        };
        let num_elements: usize = shapes.iter().product();
        let mut dense_data = vec![0.0f64; num_elements];

        gamma.0.into_iter().for_each(|(indices, value)| {
            let flat_index = indices
                .iter()
                .enumerate()
                .fold(0usize, |acc, (dim, &idx)| acc + idx * strides[dim]);
            dense_data[flat_index] = value;
        });

        let matrix = Array::from_vec(dense_data);
        let matrix = matrix
            .into_shape_with_order(shapes)
            .expect("BUG: the shapes should match the total size");
        Ok(SparseCouplingOrMatrix::Dense(matrix))
    }
}

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greedy_mec, m)?)?;
    Ok(())
}
