# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
import pathlib

import gufe
import openmm
import pytest
from gufe import ChemicalSystem, SmallMoleculeComponent
from gufe.protocols.errors import ProtocolUnitExecutionError
from openff.units import unit

import openfe
from openfe.data._registry import POOCH_CACHE
from openfe.protocols.openmm_md.plain_md_methods import PlainMDProtocol, PlainMDSimulationUnit

from ...conftest import HAS_INTERNET


@pytest.fixture()
def vacuum_protocol_settings():
    # setup a cheap vacuum md protocol
    settings = PlainMDProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    settings.engine_settings.compute_platform = None
    settings.simulation_settings.equilibration_length_nvt = 1 * unit.picoseconds
    settings.simulation_settings.equilibration_length = 1 * unit.picoseconds
    settings.simulation_settings.production_length = 1 * unit.picoseconds
    settings.output_settings.checkpoint_interval = 0.5 * unit.picoseconds
    settings.output_settings.trajectory_write_interval = 0.5 * unit.picoseconds
    return settings


def test_verify_execution_environment():
    # verify using the current versions of the software
    PlainMDSimulationUnit._verify_execution_environment(
        setup_outputs={
            "gufe_version": gufe.__version__,
            "openfe_version": openfe.__version__,
            "openmm_version": openmm.__version__,
        }
    )


def test_verify_execution_environment_fail():
    # pass in different versions to force failure
    with pytest.raises(ProtocolUnitExecutionError, match="Python environment"):
        PlainMDSimulationUnit._verify_execution_environment(
            setup_outputs={
                "gufe_version": 0.1,
                "openfe_version": openmm.__version__,
                "openmm_version": openmm.__version__,
            }
        )


def test_verify_execution_env_missing_key():
    errmsg = "Missing environment information from setup outputs."
    with pytest.raises(ProtocolUnitExecutionError, match=errmsg):
        PlainMDSimulationUnit._verify_execution_environment(
            setup_outputs={
                "foo_version": 0.1,
                "openfe_version": openfe.__version__,
                "openmm_version": openmm.__version__,
            },
        )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet unavailable and test data is not cached locally",
)
def test_check_restart(vacuum_protocol_settings):
    # test we can correctly detect when we should be restarting

    # make sure it does not try and restart if inputs are missing
    assert not PlainMDSimulationUnit._check_restart(
        output_settings=vacuum_protocol_settings.output_settings,
        shared_path=pathlib.Path("."),
    )
