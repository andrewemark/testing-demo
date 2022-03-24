from .affine import Transform2D
from .point import Point

import numpy.testing as npt
import pytest


class TestPoint:
    def test_construct(self):
        p = Point(2.0, -3.0)
        assert p.x == 2.0
        assert p.y == -3.0

    def test_add(self):
        p = Point(2.0, -3.0)
        result = p + Point(1.11, -1.12)

        # Bad floating point comparison!
        # assert result.x == 3.11
        # assert result.y == -1.22

        # A numpy test function version
        npt.assert_allclose(result.x, 3.11)
        npt.assert_allclose(result.y, -4.12)

        # A pytest version
        assert result.x == pytest.approx(3.11)
        assert result.y == pytest.approx(-4.12)


    def test_inplace_add(self):
        ...

    def test_transform_point(self):
        p = Point(1.0, -1.0)
        trans = Transform2D().translate(-1.0, 1.0)
        result = p.transform(trans, in_place=False)
        npt.assert_allclose(result.x, 0.0)
        npt.assert_allclose(result.y, 0.0)

        # Not an inplace transformation
        assert id(p) != id(result)

    def test_transform_inplace(self):
        p = Point(1.0, -1.0)
        trans = Transform2D().translate(-1.0, 1.0)
        result = p.transform(trans, in_place=True)
        assert result is None
        npt.assert_allclose(p.x, 0.0)
        npt.assert_allclose(p.y, 0.0)
