# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import warnings
with warnings.catch_warnings():
    # openff complains about oechem being missing, shhh
    from openff.toolkit.topology import Molecule as OFFMolecule

from rdkit import Chem

from openfe.utils.molhashing import hashmol
from openfe.utils.typing import RDKitMol, OEMol


class Molecule:
    """Molecule wrapper to provide proper hashing and equality.

    This class is a read-only representation of a molecule, if you want
    to edit the molecule do this in an appropriate toolkit **before** creating
    this class.

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
        # internally we store as RDKit, for now...
        self._rdkit = Chem.Mol(rdkit)
        self._hash = hashmol(self._rdkit, name=name)

    def to_rdkit(self) -> RDKitMol:
        """Return an RDKit copied representation of this molecule"""
        return Chem.Mol(self._rdkit)

    @classmethod
    def from_rdkit(cls, rdkit: RDKitMol, name: str = ""):
        """Create a Molecule copying the input from an rdkit Mol"""
        return cls(rdkit=Chem.Mol(rdkit), name=name)

    def to_openeye(self) -> OEMol:
        """OEChem representation of this molecule"""
        return self.to_openff().to_openeye()

    @classmethod
    def from_openeye(cls, oemol: OEMol, name: str = ""):
        raise NotImplementedError

    def to_openff(self) -> OFFMolecule:
        """OpenFF Toolkit representation of this molecule"""
        m = OFFMolecule(self._rdkit, allow_undefined_stereo=True)
        m.name = self.name

        return m

    @classmethod
    def from_openff(cls, openff: OFFMolecule, name: str = ""):
        raise NotImplementedError

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
