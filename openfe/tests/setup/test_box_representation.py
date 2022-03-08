# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import numpy as np
import pytest

from openfe.setup import BoxRepresentation


@pytest.fixture
def triclinic_box():
    return BoxRepresentation(np.array([[1, 0, 0], [0.2, 1, 0], [0.2, 0.2, 1]]))


@pytest.fixture
def nearly_triclinic_box(triclinic_box):
    # oh so close, but not quite
    alt_boxarr = np.array(triclinic_box.to_matrix())
    alt_boxarr[0][0] += np.nextafter(1, 2, dtype=np.float64)  # smallest delta
    return BoxRepresentation(alt_boxarr)


@pytest.fixture
def ref_bytes(triclinic_box):
    return triclinic_box.to_matrix().tobytes(order='C')


def test_to_matrix(triclinic_box):
    arr = triclinic_box.to_matrix()

    assert isinstance(arr, np.ndarray)


def test_box_eq(triclinic_box):
    box2 = BoxRepresentation(triclinic_box.to_matrix())

    assert triclinic_box == box2


def test_box_neq(triclinic_box, nearly_triclinic_box):
    assert triclinic_box != nearly_triclinic_box


def test_box_neq2(triclinic_box):
    assert triclinic_box != triclinic_box.to_matrix()


def test_box_hash(triclinic_box):
    box2 = BoxRepresentation(triclinic_box.to_matrix())

    assert hash(box2) == hash(triclinic_box)


def test_box_nhash(triclinic_box, nearly_triclinic_box):
    assert hash(triclinic_box) != hash(nearly_triclinic_box)


def test_to_bytes(triclinic_box):
    val = triclinic_box.to_bytes()

    assert isinstance(val, bytes)
    assert len(val) == 9 * 8


def test_from_bytes(triclinic_box, ref_bytes):
    newbox = BoxRepresentation.from_bytes(ref_bytes)

    assert newbox == triclinic_box


def test_invalid_box():
    arr = np.eye(3)
    arr[0][1] = 1.0
    with pytest.raises(ValueError):
        _ = BoxRepresentation(arr)


def test_invalid_box2():
    arr = np.array([10, 10, 10, 90, 90, 90])

    with pytest.raises(ValueError):
        _ = BoxRepresentation(arr)
