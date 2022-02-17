# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from openff.toolkit.topology import Molecule as OFFMolecule
from typing import TypeVar

from openfe.utils.molhashing import hashmol

RDKitMol = TypeVar('RDKitMol')
OEMol = TypeVar('OEMol')


class Molecule:
    """Molecule wrapper to provide proper hashing and equality.

    Parameters
    ----------
    rdkit : rdkit.Mol
        rdkit representation of the molecule
    name : str, optional
        if multiple Molecules with identical SMILES but differing positions
        are used, a name must be given to differentiate these.  This name
        will be used in the hash.
    """
    def __init__(self, rdkit: RDKitMol, name: str = ""):
        self._rdkit = rdkit
        self._hash = hashmol(self._rdkit, name=name)

    # property for immutability; also may allow in-class type conversion
    @property
    def rdkit(self) -> RDKitMol:
        """RDKit representation of this molecule"""
        return self._rdkit

    @property
    def oechem(self) -> OEMol:
        """OEChem representation of this molecule"""
        return self.openff.to_openeye()

    @property
    def openff(self) -> OFFMolecule:
        """OpenFF Toolkit representation of this molecule"""
        return OFFMolecule(self.rdkit, allow_undefined_stereo=True)

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
