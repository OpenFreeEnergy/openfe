# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openfe.utils.molhashing import hashmol

class Molecule:
    """Molecule wrapper to provide proper hashing and equality.

    Parameters
    ----------
    rdkit : rdkit.Mol
        rdkit representation of the molecule
    """
    def __init__(self, rdkit):
        self._rdkit = rdkit
        self._hash = None

    # property for immutability; also may allow in-class type conversion
    @property
    def rdkit(self):
        return self._rdkit

    def __hash__(self):
        if self._hash is None:
            self._hash = hashmol(self.rdkit)
        return hash(self._hash)

    def __eq__(self, other):
        return hash(self) == hash(other)
