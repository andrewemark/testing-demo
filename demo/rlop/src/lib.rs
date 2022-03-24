//! Demo Rusty linear operator library.
trait LinearOperator {
    fn forward(&self, x: &[f64]) -> Vec<f64>;
    fn adjoint(&self, y: &[f64]) -> Vec<f64>;
}

#[repr(C)]
pub struct DenseMatrix {
    num_rows: usize,
    num_cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    fn new(data: &[f64], num_rows: usize, num_cols: usize) -> Self {
        Self {
            data: data.to_vec(),
            num_rows,
            num_cols,
        }
    }
}

fn dot_product(num_rows: usize, num_cols: usize, mat: &[f64], x: &[f64]) -> Vec<f64> {
    debug_assert!(num_rows * num_cols == mat.len());
    let mut y = vec![0.0; num_rows];
    let mut offset: usize = 0;
    for ridx in 0..num_rows {
        for cidx in 0..num_cols {
            y[ridx] += mat[offset + cidx] * x[cidx];
        }
        offset += num_cols;
    }
    y
}

fn transpose(num_rows: usize, num_cols: usize, mat: &[f64]) -> Vec<f64> {
    debug_assert!(num_rows * num_cols == mat.len());
    let mut tmat: Vec<f64> = Vec::with_capacity(mat.len());

    for cidx in 0..num_cols {
        for ridx in 0..num_rows {
            tmat.push(mat[cidx + ridx * num_cols]);
        }
    }
    tmat
}

impl LinearOperator for DenseMatrix {
    fn forward(&self, x: &[f64]) -> Vec<f64> {
        dot_product(self.num_rows, self.num_cols, &self.data, x)
    }
    fn adjoint(&self, y: &[f64]) -> Vec<f64> {
        let tmat = transpose(self.num_rows, self.num_cols, &self.data);
        dot_product(self.num_cols, self.num_rows, &tmat, y)
    }
}

// --------------------------------------------------------------------------------------------- //
//                                    Exposed C API                                              //
// --------------------------------------------------------------------------------------------- //

#[no_mangle]
pub extern "C" fn dense_matrix_new(
    data: *const f64,
    num_rows: usize,
    num_cols: usize,
) -> *mut DenseMatrix {
    let row_major_matrix = unsafe { std::slice::from_raw_parts(data, num_rows * num_cols) };
    let lop = DenseMatrix::new(row_major_matrix, num_rows, num_cols);

    Box::into_raw(Box::new(lop))
}

#[no_mangle]
pub extern "C" fn dense_matrix_delete(dmat: *mut DenseMatrix) {
    if dmat.is_null() {
        return;
    }

    unsafe {
        Box::from_raw(dmat);
    }
}

#[no_mangle]
pub extern "C" fn dense_matrix_forward(dmat: *const DenseMatrix, x: *const f64, y: *mut f64) {
    let dmat: &DenseMatrix = unsafe { &*dmat };
    let x = unsafe { std::slice::from_raw_parts(x, dmat.num_cols) };
    let y: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(y, dmat.num_rows) };

    let y_tmp = dmat.forward(x);
    y.copy_from_slice(&y_tmp);
}

#[no_mangle]
pub extern "C" fn dense_matrix_adjoint(dmat: *const DenseMatrix, y: *const f64, x: *mut f64) {
    let dmat: &DenseMatrix = unsafe { &*dmat };
    let y = unsafe { std::slice::from_raw_parts(y, dmat.num_rows) };
    let x: &mut [f64] = unsafe { std::slice::from_raw_parts_mut(x, dmat.num_cols) };

    let x_tmp = dmat.adjoint(y);
    x.copy_from_slice(&x_tmp);
}

#[cfg(test)]
mod tests {
    use crate::{DenseMatrix, LinearOperator};

    #[test]
    fn test_dense_forward() {
        let mat: Vec<f64> = vec![1., 2., 3., 4., 5., 6.];
        let x: Vec<f64> = vec![1.7, 2.2, 3.9];

        let dense_mat = DenseMatrix::new(&mat, 2, 3);
        let y_exp = vec![17.8, 41.2];
        let y = dense_mat.forward(&x);

        for (&v_exp, &v) in y_exp.iter().zip(y.iter()) {
            assert!((v_exp - v).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dense_adjoint() {
        let mat: Vec<f64> = vec![1., 2., 3., 4., 5., 6.];
        let y: Vec<f64> = vec![6.1, 0.4];
        let dense_mat = DenseMatrix::new(&mat, 2, 3);
        let x_exp = vec![7.7, 14.2, 20.7];
        let x = dense_mat.adjoint(&y);

        for (&v_exp, &v) in x_exp.iter().zip(x.iter()) {
            assert!((v_exp - v).abs() < 1e-8);
        }
    }
}
