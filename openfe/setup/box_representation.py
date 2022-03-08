# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import numpy as np


def is_valid_box(box: np.ndarray) -> bool:
    """Check that a box is valid

    Current rules:
    - the y and z component of the x vector must be zero
    - the z component of the y vector must be zero

    Parameters
    ----------
    box : np.ndarray, shape(3,3)
       the simulation cell

    Returns
    -------
    valid : bool
    """
    if box.shape != (3, 3):
        return False
    lx, xy, xz = box[0]
    yx, ly, yz = box[1]
    zx, zy, lz = box[2]

    if (xy != 0.0) or (xz != 0.0) or (yz != 0.0):
        return False
    # TODO: Max tilt factors, are these common across all MD engines?
    return True


class BoxRepresentation:
    """The simulation cell in which all components sit

    For the purposes of simulation planning, the box is considered a component
    of the microstate.

    Internally this will be stored as float64 values (aka doubles).
    """
    def __init__(self, box):
        """
        Parameters
        ----------
        box : np.ndarray, shape(3,3)
          the 3x3 matrix representation of the box.  The first line is the
          vector parallel to the "x" dimension, then the vector mostly parallel
          to "y" then "z".

        Notes
        -----
        The x vector must be parallel to the x axis, i.e. the [0][1] and [0][2]
        values must be zero.  The y vector must be orthogonal to the z axis,
        i.e. the [1][2] value must be zero

        Raises
        ------
        ValueError
          if the above constraints are not met
        """
        self._box = np.asarray(box, dtype=np.float64, order='C')

        if not is_valid_box(self._box):
            raise ValueError("Box was not valid")

    @classmethod
    def from_bytes(cls, serialisation: bytes):
        """Deserialise a Box

        Examples
        --------
        The from_str classmethod is the inverse operation to str, i.e.:

          box = BoxRepresentation(np.eye(3))
          newbox = BoxRepresentation.from_str(str(box))

          assert newbox == newbox
        """
        return cls(np.frombuffer(serialisation, dtype=np.float64))

    def to_bytes(self) -> bytes:
        """A byte based representation for serialisation

        This is the 3x3 matrix, expressed as float64 (doubles) in C order

        This allows a lossless serialisation mechanism for the box.
        """
        return self.to_matrix().tobytes(order='C')

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.to_matrix() == other.to_matrix()).all()

    def __hash__(self):
        return hash(self.to_bytes())

    def to_matrix(self) -> np.ndarray:
        """Returns a 3x3 matrix of box vectors"""
        return np.array(self._box)  # return copy
