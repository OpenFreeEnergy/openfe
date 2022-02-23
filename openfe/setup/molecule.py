# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import contextlib
import io
import warnings
from typing import TypeVar

from rdkit import Chem

import openfe
from openfe.utils.molhashing import hashmol

RDKitMol = TypeVar('RDKitMol')


def _ensure_ofe_name(mol, name):
    """
    Ensure that rdkit mol carries the ofe-name and ofe-version tags.
    """
    rdkit_name = name  # override this if the property is defined
    if name == "":
        with contextlib.suppress(KeyError):
            rdkit_name = mol.GetProp("ofe-name")

    if name != "" and rdkit_name != name:
        warnings.warn(f"Molecule being renamed from {rdkit_name} to {name}.")
    elif name == "":
        name = rdkit_name

    mol.SetProp("ofe-name", name)
    return name


def _ensure_ofe_version(mol):
    mol.SetProp("ofe-version", openfe.__version__)


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
        name = _ensure_ofe_name(rdkit, name)
        _ensure_ofe_version(rdkit)
        self._rdkit = rdkit
        self._hash = hashmol(self._rdkit, name=name)

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

    def to_sdf(self) -> str:
        # https://sourceforge.net/p/rdkit/mailman/message/27518272/
        mol = self.rdkit
        sdf = [Chem.MolToMolBlock(mol)]
        for prop in mol.GetPropNames():
            # always output as this version of OpenFE
            if prop == "ofe-version":
                val = openfe.__version__
            else:
                val = mol.GetProp(prop)
            sdf.append('>  <%s>\n%s\n' % (prop, val))
        sdf.append('$$$$\n')
        return "\n".join(sdf)

    @classmethod
    def from_sdf_string(cls, sdf_str: str):
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
