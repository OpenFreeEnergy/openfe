# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import TypeVar
RDKitMol = TypeVar('RDKitMol')

from openfe.utils.molhashing import hashmol


class Molecule:
    """Molecule wrapper to provide proper hashing and equality.

    Parameters
    ----------
    rdkit : rdkit.Mol
        rdkit representation of the molecule
    """
    def __init__(self, rdkit: RDKitMol):
        self._rdkit = rdkit
        self._hash = hashmol(self._rdkit)

    # property for immutability; also may allow in-class type conversion
    @property
    def rdkit(self) -> RDKitMol:
        """RDKit representation of this molecule"""
        return self._rdkit

    @property
    def smiles(self):
        return self._hash.smiles

    @property
    def name(self):
        return self._hash.name

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        return hash(self) == hash(other)
