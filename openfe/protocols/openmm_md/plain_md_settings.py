# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for plain MD Protocols using OpenMM + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.openmm_md.plain_md_methods.py`

"""
from openfe.protocols.openmm_utils.omm_settings import (
    Settings, SystemSettings,
    SolvationSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings, RepeatSettings
)


class PlainMDProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    # Things for creating the systems
    system_settings: SystemSettings
    solvation_settings: SolvationSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: SimulationSettings

    # Setting number of repeats of md simulation
    repeat_settings : RepeatSettings