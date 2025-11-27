# Before we do anything else, we want to disable JAX
# acceleration by default but if a user has set
# PYMBAR_DISABLE_JAX to some value, we want to keep
# it

import os

os.environ.setdefault("PYMBAR_DISABLE_JAX", "TRUE")

# We need to do this first so that we can set up our
# log control since some modules have warnings on import
from openfe.utils import logging_control

logging_control._silence_message(
    msg=[
        "****** PyMBAR will use 64-bit JAX! *******",
    ],
    logger_names=[
        "pymbar.mbar_solvers",
    ],
)

logging_control._silence_message(
    msg=[
        "Warning on use of the timeseries module:",
    ],
    logger_names=[
        "pymbar.timeseries",
    ],
)

logging_control._append_logger(
    suffix="\n \n[OPENFE]: The simulation is still using the compute platform specified in the settings \n See this URL for more information: https://docs.openfree.energy/en/latest/guide/troubleshooting.html#jax-warnings \n\n",
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
