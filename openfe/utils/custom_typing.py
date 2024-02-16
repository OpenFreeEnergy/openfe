# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import TypeVar

import matplotlib.axes
import matplotlib.backend_bases
from rdkit import Chem

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias

RDKitMol: TypeAlias = Chem.rdchem.Mol

OEMol = TypeVar("OEMol")
MPL_FigureCanvasBase: TypeAlias = matplotlib.backend_bases.FigureCanvasBase
MPL_MouseEvent: TypeAlias = matplotlib.backend_bases.MouseEvent
MPL_Axes: TypeAlias = matplotlib.axes.Axes
