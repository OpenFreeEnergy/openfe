# We need to do this first so that we can set up our
# log control since some modules have warnings on import
from openfe.utils.logging_control import LogControl


LogControl.silence_message(
    msg=["****** PyMBAR will use 64-bit JAX! *******",],
    logger_names=["pymbar.mbar_solvers",]
)

LogControl.silence_message(
    msg=["Warning on use of the timeseries module:",],
    logger_names=["pymbar.timeseries",]
)

LogControl.append_logger(
        suffix="\n \n[OPENFE]: See this url for more information about the warning above\n",
         logger_names="jax._src.xla_bridge",
         )

# These two lines are just to test the append_logger and will be removed before
# the PR is merged
from jax._src.xla_bridge import backends
backends()

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
    ProtocolUnitResult,
    ProtocolUnitFailure,
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
