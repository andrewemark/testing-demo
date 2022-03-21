import numpy as np
from dataclasses import dataclass


def identity():
    return np.identity(3)


def scale(x: float = 1.0, y: float = 1.0):
    return np.array(
        [
            [x, 0, 0],
            [0, y, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


def translate(x: float = 0.0, y: float = 0.0):
    return np.array(
        [
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        ],
        dtype=float,
    )


def reflect(x: bool = False, y: bool = False):
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
