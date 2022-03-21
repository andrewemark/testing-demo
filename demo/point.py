from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def transform(self, tmat: np.ndarray, in_place=False):
        x_new, y_new, _ = tmat @ np.array([self.x, self.y, 1]).T

        if in_place:
            self.x = x_new
            self.y = y_new
            return None

        return Point(x_new, y_new)
