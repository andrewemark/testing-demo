import pytest

from demo.affine import Transform2D
import numpy.testing as npt
import numpy as np


def test_identity():
    t = Transform2D()
    res = t.transform_point(5.0, -2.0)
    npt.assert_allclose(res, [5.0, -2.0])


def test_translate():
    t = Transform2D().translate(1.0, 2.0)
    res = t.transform_point(0.0, 0.0)
    npt.assert_allclose(res, [1.0, 2.0])


def test_scale():
    t = Transform2D().scale(1.0, 2.0)
    res = t.transform_point(3.0, -1.0)
    npt.assert_allclose(res, [3.0, -2.0])


def test_compose_transforms():
    x = 2.0
    y = 3.0

    t = Transform2D().translate(1.0, 4.0).scale(0.5, 2.0)

    expected_x_hat = 0.5 * (x + 1.0)
    expected_y_hat = 2.0 * (y + 4.0)

    x_hat, y_hat = t.transform_point(x, y)

    npt.assert_allclose([expected_x_hat, expected_y_hat], [x_hat, y_hat])


def test_to_numpy():
    t = Transform2D()
    npt.assert_allclose(t.to_numpy(), np.identity(3))


def test_construct_from_numpy():
    tmat = np.array(
        [
            [1, 0, 2],
            [0, -1, 0],
            [0, 0, 1],
        ]
    ).astype(float)

    t = Transform2D(tmat)

    npt.assert_array_equal(tmat, t.to_numpy())


def test_invalid_construct_from_numpy():
    tmat = np.array(
        [
            [1, 0, 2],
            [0, -1, 0],
        ]
    ).astype(float)

    with pytest.raises(ValueError):
        Transform2D(tmat)
