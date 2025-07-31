# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for plain MD Protocols using OpenMM + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.openmm_md.plain_md_methods.py`

"""
from typing import Annotated
from annotated_types import Gt
from openfe.protocols.openmm_utils.omm_settings import (
    Settings,
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    MDSimulationSettings,
    IntegratorSettings, MDOutputSettings,
    OpenFFPartialChargeSettings,
)
from gufe.settings import (
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings
)

class PlainMDProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    protocol_repeats: Annotated[int, Gt(0)]
    """
    Number of independent MD runs to perform.
    """

    # Things for creating the systems
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    partial_charge_settings: OpenFFPartialChargeSettings
    solvation_settings: OpenMMSolvationSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: MDSimulationSettings

    # Simulations output settings
    output_settings: MDOutputSettings
