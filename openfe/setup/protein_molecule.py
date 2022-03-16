# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem

from openfe.utils.typing import RDKitMol, OEMol


class ProteinMolecule:
    """Wrapper around a Protein representation

    This representation is immutable.  If you want to make any modifications,
    do this in an appropriate toolkit then remake this class.
    """
    def __init__(self, rdkit: RDKitMol, name=""):
        self._rdkit = rdkit
        self._name = name

    @property
    def name(self):
        return self._name

    @classmethod
    def from_pdbfile(cls, pdbfile: str, name=""):
        cls(rdkit=Chem.MolFromPDBFile(pdbfile), name=name)

    @classmethod
    def from_pdbxfile(cls, pdbxfile: str, name=""):
        raise NotImplementedError()

    @classmethod
    def from_rdkit(cls, rdkit: RDKitMol, name=""):
        return cls(rdkit=Chem.Mol(rdkit), name=name)

    def to_rdkit(self) -> RDKitMol:
        return Chem.Mol(self._rdkit)

    @classmethod
    def from_openff(cls, offmol, name=""):
        raise NotImplementedError()

    def to_openff(self):
        raise NotImplementedError()

    @classmethod
    def from_openeye(cls, oemol: OEMol, name=""):
        raise NotImplementedError()

    def to_openeye(self) -> OEMol:
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.name, Chem.MolToSequence(self._rdkit))

    def __eq__(self, other):
        return hash(self) == hash(other)
