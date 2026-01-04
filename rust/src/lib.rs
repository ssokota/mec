use numpy::{
    convert::IntoPyArray,
    ndarray::{Array, Dim, IxDynImpl},
    PyArray1, PyArrayMethods, PyReadonlyArray1,
};
use pyo3::{prelude::*, types::PyTuple};
mod sparse_coupling;
use sparse_coupling::SparseCoupling;

const TOLERANCE: f64 = 1e-9;
const RTOL: f64 = 1e-5;  // Relative tolerance matching np.isclose
const ATOL: f64 = 1e-8;  // Absolute tolerance matching np.isclose

/// Check if two values are close (matching np.isclose behavior)
#[inline]
fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * b.abs()
}

/// Calculate Shannon entropy of a probability distribution
#[pyfunction]
fn entropy(py: Python, distribution: &Bound<'_, PyAny>) -> PyResult<f64> {
    // Try to get the array as a contiguous f64 array
    let np = py.import("numpy")?;
    let arr_flat = np.call_method1("asarray", (distribution,))?.call_method1("flatten", ())?;
    let arr = arr_flat.downcast::<PyArray1<f64>>()?;
    let readonly = arr.readonly();
    let view = readonly.as_array();

    let mut result = 0.0;
    for &p in view.iter() {
        if p > TOLERANCE {  // Skip near-zero values
            result -= p * p.ln();
        }
    }

    Ok(result)
}

/// Check if an array is a valid probability distribution
#[pyfunction]
fn is_distribution(z: PyReadonlyArray1<f64>) -> bool {
    let arr = z.as_array();
    let mut sum = 0.0;

    for &val in arr.iter() {
        // Allow small negative values if they're close to 0
        if val < 0.0 && !is_close(val, 0.0) {
            return false;
        }
        if !val.is_finite() {
            return false;  // NaN or infinity not allowed
        }
        sum += val;
    }

    is_close(sum, 1.0)
}

/// Check if a distribution is deterministic (all mass on one state)
#[pyfunction]
fn is_deterministic(z: PyReadonlyArray1<f64>) -> bool {
    let arr = z.as_array();
    let mut found_one = false;

    for &val in arr.iter() {
        if is_close(val, 1.0) {
            if found_one {
                return false;  // More than one value close to 1
            }
            found_one = true;
        } else if val > 0.0 && !is_close(val, 0.0) {
            return false;  // Value that's not 0 or 1
        }
    }

    found_one
}

/// Compute upper bounds on entropies of distributions
#[pyfunction]
fn entropy_upper_bounds(z: PyReadonlyArray1<f64>, num_states: usize) -> Py<PyArray1<f64>> {
    let arr = z.as_array();
    let log_num_states = (num_states as f64).ln();
    let uniform_prob = 1.0 / (num_states as f64);

    let mut result = Vec::with_capacity(arr.len());

    for &z_val in arr.iter() {
        let mut upper_bound = log_num_states;

        // If lower bound greater than uniform, can beat trivial upper bound
        if z_val > uniform_prob {
            upper_bound = -z_val * z_val.ln();
            let remaining_mass = 1.0 - z_val;

            // Only add remaining mass contribution if there's significant mass left
            if remaining_mass > TOLERANCE {
                let remaining_mass_per_state = remaining_mass / ((num_states - 1) as f64);
                upper_bound -= remaining_mass * remaining_mass_per_state.ln();
            }
        }

        result.push(upper_bound);
    }

    Python::with_gil(|py| Array::from_vec(result).into_pyarray(py).unbind())
}

/// Find rows in matrix proportional to given row
#[pyfunction]
fn get_proportional_rows(
    matrix_flat: PyReadonlyArray1<f64>,
    row_index: usize,
    num_cols: usize,
) -> Py<PyArray1<i64>> {
    let flat = matrix_flat.as_array();
    let num_rows = flat.len() / num_cols;

    // Get the reference row
    let ref_row_start = row_index * num_cols;
    let ref_row = &flat.as_slice().unwrap()[ref_row_start..ref_row_start + num_cols];

    // Find non-zero pattern of reference row
    let ref_non_zero: Vec<bool> = ref_row.iter().map(|&x| x > 0.0).collect();
    let ref_non_zero_indices: Vec<usize> = ref_non_zero
        .iter()
        .enumerate()
        .filter(|(_, &nz)| nz)
        .map(|(i, _)| i)
        .collect();

    if ref_non_zero_indices.is_empty() {
        // If reference row is all zeros, return just this row
        return Python::with_gil(|py| {
            Array::from_vec(vec![row_index as i64])
                .into_pyarray(py)
                .unbind()
        });
    }

    // Normalize reference row (only non-zero entries)
    let ref_sum: f64 = ref_non_zero_indices.iter().map(|&i| ref_row[i]).sum();
    let ref_normalized: Vec<f64> = ref_non_zero_indices
        .iter()
        .map(|&i| ref_row[i] / ref_sum)
        .collect();

    let mut proportional_indices = Vec::new();

    for row_idx in 0..num_rows {
        let row_start = row_idx * num_cols;
        let row = &flat.as_slice().unwrap()[row_start..row_start + num_cols];

        // Check if non-zero pattern matches
        let matches_pattern = ref_non_zero
            .iter()
            .enumerate()
            .all(|(i, &ref_nz)| (row[i] > 0.0) == ref_nz);

        if !matches_pattern {
            continue;
        }

        // Normalize this row and compare
        let row_sum: f64 = ref_non_zero_indices.iter().map(|&i| row[i]).sum();

        if row_sum < TOLERANCE {
            continue;
        }

        let is_proportional = ref_non_zero_indices
            .iter()
            .zip(&ref_normalized)
            .all(|(&i, &ref_val)| {
                let row_val = row[i] / row_sum;
                is_close(row_val, ref_val)
            });

        if is_proportional {
            proportional_indices.push(row_idx as i64);
        }
    }

    Python::with_gil(|py| {
        Array::from_vec(proportional_indices)
            .into_pyarray(py)
            .unbind()
    })
}

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
fn mec_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greedy_mec, m)?)?;
    m.add_function(wrap_pyfunction!(entropy, m)?)?;
    m.add_function(wrap_pyfunction!(is_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(is_deterministic, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_upper_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(get_proportional_rows, m)?)?;
    Ok(())
}
