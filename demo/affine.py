from typing import Optional

import numpy as np


def _identity():
    return np.identity(3)


def _scale(x: float = 1.0, y: float = 1.0):
    return np.array(
        [
            [x, 0, 0],
            [0, y, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


def _translate(x: float = 0.0, y: float = 0.0):
    return np.array(
        [
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        ],
        dtype=float,
    )


def _reflect(x: bool = False, y: bool = False):
    rx = -1 if x else 1
    ry = -1 if y else 1
    return np.array(
        [
            [rx, 0, 0],
            [0, ry, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


class Transform2D:
    def __init__(self, M: Optional[np.ndarray] = None):
        self._M = M if M is not None else _identity()
        if self._M.shape != (3, 3):
            raise ValueError("2D Transformation matrix must be 3x3")

    def scale(self, sx: float = 1.0, sy: float = 1.0):
        M_scale = _scale(sx, sy)
        self._M = M_scale @ self._M
        return self

    def translate(self, tx: float = 0.0, ty: float = 0.0):
        M_translate = _translate(tx, ty)
        self._M = M_translate @ self._M
        return self

    def inverse(self):
        return Transform2D(np.linalg.inv(self._M))

    def transform_point(self, x, y):
        return (self._M @ np.array([x, y, 1]).T)[:2]

    def to_numpy(self):
        return self._M.copy()
