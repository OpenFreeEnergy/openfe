"""
Open source free energy calculation via molecular mechanics.
"""

from .protocols import (
    Protocol,
    ProtocolDAG,
    ProtocolUnit,
    ProtocolUnitResult, ProtocolUnitFailure,
    ProtocolDAGResult,
    ProtocolResult,
)
from .orchestration import execute_DAG

from . import utils
from . import setup
from .setup import (
    LomapAtomMapper,
    lomap_scorers,
    PersesAtomMapper,
    perses_scorers,
    ligand_network_planning,
    LigandNetwork,
    LigandAtomMapper,
    LigandAtomMapping,
    Transformation,
    AlchemicalNetwork,
)
from .setup.system import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from . import orchestration
from . import analysis

from importlib.metadata import version
__version__ = version("openfe")
