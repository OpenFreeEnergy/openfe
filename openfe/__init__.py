# silence pymbar logging warnings
import logging


def _mute_timeseries(record):
    return "Warning on use of the timeseries module:" not in record.msg


def _mute_jax(record):
    return "****** PyMBAR will use 64-bit JAX! *******" not in record.msg


_mbar_log = logging.getLogger("pymbar.timeseries")
_mbar_log.addFilter(_mute_timeseries)
_mbar_log = logging.getLogger("pymbar.mbar_solvers")
_mbar_log.addFilter(_mute_jax)

from importlib.metadata import version

from gufe import (
    AlchemicalNetwork,
    ChemicalSystem,
    Component,
    LigandAtomMapping,
    NonTransformation,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    Transformation,
)
from gufe.protocols import (
    Protocol,
    ProtocolDAG,
    ProtocolDAGResult,
    ProtocolResult,
    ProtocolUnit,
    ProtocolUnitFailure,
    ProtocolUnitResult,
    execute_DAG,
)

from . import analysis, orchestration, setup, utils
from .setup import (
    LigandAtomMapper,
    LigandNetwork,
    LomapAtomMapper,
    PersesAtomMapper,
    ligand_network_planning,
    lomap_scorers,
    perses_scorers,
)

__version__ = version("openfe")
