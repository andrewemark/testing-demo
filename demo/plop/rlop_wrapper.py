import os
import ctypes
import platform

import numpy as np
import numpy.typing as npt


class OpaqueDenseMatrix(ctypes.Structure):
    pass


plat = platform.system()
if plat == "Windows":
    LIBEXT = ".dll"
elif plat == "Darwin":
    LIBEXT = ".dylib"
else:
    LIBEXT = ".so"

librlop = np.ctypeslib.load_library(
    libname=f"librlop{LIBEXT}",
    loader_path=os.path.abspath(f"{os.path.dirname(__file__)}/../rlop/target/release"),
)

librlop.dense_matrix_new.restype = ctypes.POINTER(OpaqueDenseMatrix)
librlop.dense_matrix_new.argstype = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
]

librlop.dense_matrix_delete.argstype = [
    ctypes.POINTER(OpaqueDenseMatrix),
]

librlop.dense_matrix_forward.argstype = [
    ctypes.POINTER(OpaqueDenseMatrix),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]

librlop.dense_matrix_adjoint.argstype = [
    ctypes.POINTER(OpaqueDenseMatrix),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def new_dense_matrix(mat: npt.NDArray[np.float64]) -> ctypes.POINTER(OpaqueDenseMatrix):
    num_rows, num_cols = mat.shape
    data = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    num_rows = ctypes.c_size_t(num_rows)
    num_cols = ctypes.c_size_t(num_cols)

    return librlop.dense_matrix_new(data, num_rows, num_cols)


def forward(
    dmat: ctypes.POINTER(OpaqueDenseMatrix),
    num_rows: int,
    num_cols: int,
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    y = np.zeros(num_rows, dtype=np.float64)
    x_raw = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_raw = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    librlop.dense_matrix_forward(dmat, x_raw, y_raw)

    return y


def adjoint(
    dmat: ctypes.POINTER(OpaqueDenseMatrix),
    num_rows: int,
    num_cols: int,
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    x = np.zeros(num_cols, dtype=np.float64)
    y_raw = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    x_raw = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    librlop.dense_matrix_adjoint(dmat, y_raw, x_raw)

    return x


def free(dmat: ctypes.POINTER(OpaqueDenseMatrix)):
    librlop.dense_matrix_delete(dmat)
