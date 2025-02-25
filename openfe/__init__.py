# silence pymbar logging warnings
import logging
def _mute_timeseries(record):
    return not "Warning on use of the timeseries module:" in record.msg
def _mute_jax(record):
    return not "****** PyMBAR will use 64-bit JAX! *******" in record.msg
_mbar_log = logging.getLogger("pymbar.timeseries")
_mbar_log.addFilter(_mute_timeseries)
_mbar_log = logging.getLogger("pymbar.mbar_solvers")
_mbar_log.addFilter(_mute_jax)

from gufe import (
    ChemicalSystem,
    Component,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
    NonTransformation,
    AlchemicalNetwork,
    LigandAtomMapping,
)
from gufe.protocols import (
    Protocol,
    ProtocolDAG,
    ProtocolUnit,
    ProtocolUnitResult, ProtocolUnitFailure,
    ProtocolDAGResult,
    ProtocolResult,
    execute_DAG,
)

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
)
from . import orchestration
from . import analysis

from importlib.metadata import version
__version__ = version("openfe")
