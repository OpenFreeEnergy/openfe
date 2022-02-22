# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import io
from typing import TypeVar
RDKitMol = TypeVar('RDKitMol')

from rdkit import Chem

from openfe.utils.molhashing import hashmol
import openfe


def _ensure_ofe_name(mol, name):
    """
    Ensure that rdkit mol carries the ofe-name and ofe-version tags.
    """
    rdkit_name = name  # override this if the property is defined
    if name == "":
        try:
            rdkit_name = mol.GetProp("ofe-name")
        except KeyError:
            pass

    if name != "" and rdkit_name != name:
        raise RuntimeError(f"Internal inconsistency: 'name' has "
                           f"value '{rdkit_name}' in the internal RDKit "
                           f"representation; expected '{name}'")
    elif name == "":
        name = rdkit_name

    mol.SetProp("ofe-name", name)
    return name


def _ensure_ofe_version(mol):
    try:
        version = mol.GetProp("ofe-version")
    except KeyError:
        version = openfe.__version__
        mol.SetProp("ofe-version", version)


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
        mols = list(supp)
        assert len(mols) == 1, f"Expected 1 molecule, found {len(mols)}"
        return cls(rdkit=mols[0])  # name is obtained automatically
