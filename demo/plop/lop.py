from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import demo.plop.rlop_wrapper as rw


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def adjoint(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


@dataclass(frozen=True)
class PDenseMatrix(LinearOperator):
    inner: npt.NDArray[np.float64]

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.inner @ x

    def adjoint(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.inner.conj().T @ y


class RDenseMatrix(LinearOperator):
    def __init__(self, inner: npt.NDArray[np.float64]):
        num_rows, num_cols = inner.shape
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.obj = rw.new_dense_matrix(inner.reshape(inner.shape, order="C"))

    def free(self):
        rw.free(self.obj)

    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return rw.forward(self.obj, self.num_rows, self.num_cols, x)

    def adjoint(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return rw.adjoint(self.obj, self.num_rows, self.num_cols, y)
