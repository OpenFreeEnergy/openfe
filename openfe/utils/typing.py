# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import TypeVar
from rdkit import Chem

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

RDKitMol: TypeAlias = Chem.rdchem.Mol
OEMol = TypeVar('OEMol')
