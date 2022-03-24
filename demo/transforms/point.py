from dataclasses import dataclass

from .affine import Transform2D


@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def transform(self, trans: Transform2D, in_place=False):
        x_new, y_new = trans.transform_point(self.x, self.y)

        if in_place:
            self.x = x_new
            self.y = y_new
            return None

        return Point(x_new, y_new)
