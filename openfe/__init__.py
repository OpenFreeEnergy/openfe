# We need to do this first so that we can set up our
# log control since some modules have warnings on import
from openfe.utils.logging_control import LogControl

LogControl.silence_message(
    msg=[
        "****** PyMBAR will use 64-bit JAX! *******",
    ],
    logger_names=[
        "pymbar.mbar_solvers",
    ],
)

LogControl.silence_message(
    msg=[
        "Warning on use of the timeseries module:",
    ],
    logger_names=[
        "pymbar.timeseries",
    ],
)

LogControl.append_logger(
        suffix="\n \n[OPENFE]: The simulation is still using the compute platform specified in the settings \n See this URL for more information: \n\n",
    logger_names="jax._src.xla_bridge",
)


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
