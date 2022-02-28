# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import warnings
with warnings.catch_warnings():
    # openff complains about oechem being missing, shhh
    from openff.toolkit.topology import Molecule as OFFMolecule

import contextlib
import io
import sys
import warnings
from typing import TypeVar

from rdkit import Chem

import openfe
from openfe.utils.molhashing import hashmol
from openfe.utils.typing import RDKitMol, OEMol


def _ensure_ofe_name(mol: RDKitMol, name: str) -> str:
    """
    Determine the correct name from the rdkit.Chem.Mol and the user-provided
    name; ensure that is set in the rdkit representation.
    """
    try:
        rdkit_name = mol.GetProp("_Name")
    except KeyError:
        rdkit_name = ""

    try:
        rdkit_name = mol.GetProp("ofe-name")
    except KeyError:
        pass

    if name and rdkit_name and rdkit_name != name:
        warnings.warn(f"Molecule being renamed from {rdkit_name} to {name}.")
    elif name == "":
        name = rdkit_name

    mol.SetProp("ofe-name", name)
    return name


def _ensure_ofe_version(mol: RDKitMol):
    """Ensure the rdkit representation has the current version associated"""
    mol.SetProp("ofe-version", openfe.__version__)


class Molecule:
    """Molecule wrapper to provide proper hashing and equality.

    This class is a read-only representation of a molecule, if you want
    to edit the molecule do this in an appropriate toolkit **before** creating
    this class.

    A molecule can have a name associated with it. The name can be
    explicitly set by the ``name`` attribute, or implicitly set based on the
    tags in the input molecular representation (if supported, as with
    RDKit). If not explicitly set on creation, the molecule will first look
    for an OpenFE-specific tag ``ofe-name``, and if that doesn't exist, for
    a commonly-used naming tag (e.g., the ``_Name`` property for RDKit
    molecules). If no name is found, the empty string is used.

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
        name = _ensure_ofe_name(rdkit, name)
        _ensure_ofe_version(rdkit)
        self._rdkit = rdkit
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
    def smiles(self) -> str:
        return self._hash.smiles

    @property
    def name(self) -> str:
        return self._hash.name

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_sdf(self) -> str:
        """Create a string based on SDF.

        This is the primary serialization mechanism for this class.

        See Also
        --------
        :meth:`.from_sdf_string` : create an object from the output of this
        """
        # https://sourceforge.net/p/rdkit/mailman/message/27518272/
        mol = self.to_rdkit()
        sdf = [Chem.MolToMolBlock(mol)]
        for prop in mol.GetPropNames():
            val = mol.GetProp(prop)
            sdf.append('>  <%s>\n%s\n' % (prop, val))
        sdf.append('$$$$\n')
        return "\n".join(sdf)

    @classmethod
    def from_sdf_string(cls, sdf_str: str):
        """Create ``Molecule`` from SDF-formatted string.

        This is the primary deserialization mechanism for this class.

        Parameters
        ----------
        sdf_str : str
            input string in SDF format

        Returns
        -------
        :class:`.Molecule` :
            the deserialized molecule
        """
        # https://sourceforge.net/p/rdkit/mailman/message/27518272/
        supp = Chem.SDMolSupplier()
        supp.SetData(sdf_str)
        mol = next(supp)

        # ensure that there's only one molecule in the file
        try:
            _ = next(supp)
        except StopIteration:
            pass
        else:
            # TODO: less generic exception type here
            raise RuntimeError(f"SDF contains more than 1 molecule")

        return cls(rdkit=mol)  # name is obtained automatically
