import numpy as np
from hypothesis import given, settings, Verbosity, example, strategies as st

from . import lop


## ------------------------------------------------------------------------- ##
##                               Unit Tests                                  ##
## ------------------------------------------------------------------------- ##


def test_pmatrix_identity():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    identity = lop.PDenseMatrix(inner=np.identity(len(x)))

    assert np.allclose(x, identity.forward(x))
    assert np.allclose(x, identity.adjoint(x))


def test_pmatrix_basic():
    x_in = np.array([1.7, 2.2, 3.9], dtype=np.float64)
    y_in = np.array([6.1, 0.4], dtype=np.float64)

    A = lop.PDenseMatrix(
        inner=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    )

    y_exp = np.array([17.8, 41.2])
    y = A.forward(x_in)
    x_exp = np.array([7.7, 14.2, 20.7])
    x = A.adjoint(y_in)

    assert np.allclose(y_exp, y)
    assert np.allclose(x_exp, x)


def test_rmatrix_identity():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    identity = lop.RDenseMatrix(inner=np.identity(len(x)))

    assert np.allclose(x, identity.forward(x))
    assert np.allclose(x, identity.adjoint(x))
    identity.free()


def test_rmatrix_basic():
    x_in = np.array([1.7, 2.2, 3.9], dtype=np.float64)
    y_in = np.array([6.1, 0.4], dtype=np.float64)

    A = lop.RDenseMatrix(
        inner=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    )

    y_exp = np.array([17.8, 41.2])
    y = A.forward(x_in)
    x_exp = np.array([7.7, 14.2, 20.7])
    x = A.adjoint(y_in)

    assert np.allclose(y_exp, y)
    assert np.allclose(x_exp, x)
    A.free()


## ------------------------------------------------------------------------- ##
##                              Property Tests                               ##
## ------------------------------------------------------------------------- ##


@settings(verbosity=Verbosity.verbose)
@example(10, 10)
@given(
    st.integers(min_value=1, max_value=1_000), st.integers(min_value=1, max_value=1_000)
)
def test_for_adjoint_test_invariant_python(rows, cols):
    x = np.random.rand(cols).astype(np.float64)
    y = np.random.rand(rows).astype(np.float64)

    A = lop.PDenseMatrix(inner=np.random.rand(rows, cols).astype(np.float64))

    y_hat = A.forward(x)
    x_hat = A.adjoint(y)

    assert np.isclose(np.dot(x, x_hat), np.dot(y, y_hat))


@settings(verbosity=Verbosity.verbose)
@example(10, 10)
@given(
    st.integers(min_value=1, max_value=1_000), st.integers(min_value=1, max_value=1_000)
)
def test_for_adjoint_test_invariant_rust(rows, cols):
    x = np.random.rand(cols).astype(np.float64)
    y = np.random.rand(rows).astype(np.float64)

    A = lop.RDenseMatrix(inner=np.random.rand(rows, cols).astype(np.float64))

    y_hat = A.forward(x)
    x_hat = A.adjoint(y)

    assert np.isclose(np.dot(x, x_hat), np.dot(y, y_hat))
    A.free()


@settings(verbosity=Verbosity.verbose)
@given(
    st.integers(min_value=1, max_value=1_000), st.integers(min_value=1, max_value=1_000)
)
def test_python_and_rust_parity(rows, cols):
    x = np.random.rand(cols).astype(np.float64)
    y = np.random.rand(rows).astype(np.float64)

    inner = np.random.rand(rows, cols).astype(np.float64)

    A_python = lop.PDenseMatrix(inner=inner)
    A_rust = lop.RDenseMatrix(inner=inner)

    y_hat_python = A_python.forward(x)
    x_hat_python = A_python.adjoint(y)
    y_hat_rust = A_rust.forward(x)
    x_hat_rust = A_rust.adjoint(y)

    assert np.allclose(x_hat_python, x_hat_rust, rtol=1e-3, atol=1e-5)
    assert np.allclose(y_hat_python, y_hat_rust, rtol=1e-3, atol=1e-5)
    A_rust.free()
