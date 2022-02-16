# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from functools import wraps
from typing import TypeVar
RDKitMol = TypeVar('RDKitMol')
OEChemMol = TypeVar('OEChemMol')
OFFTkMol = TypeVar('OFFTkMol')

from openfe.utils.molhashing import hashmol


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
    def oechem(self) -> OEChemMol:
        """OEChem representation of this molecule"""
        raise NotImplementedError  # issue #24

    @property
    def openff(self) -> OFFTkMol:
        """OpenFF Toolkit representation of this molecule"""
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


def as_rdkit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        args = (a.rdkit if isinstance(a, Molecule) else a for a in args)
        return func(*args, **kwargs)

    return inner


def as_oechem(func):
    @wraps(func)
    def inner(*args, **kwargs):
        args = (a.oechem if isinstance(a, Molecule) else a for a in args)
        return func(*args, **kwargs)

    return inner
