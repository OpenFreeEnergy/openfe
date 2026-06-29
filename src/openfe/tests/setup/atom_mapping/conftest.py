# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Dict, Tuple

import lomap
import pytest
from gufe import SmallMoleculeComponent
from rdkit import Chem

from openfe import LigandAtomMapping

from ...conftest import mol_from_smiles
