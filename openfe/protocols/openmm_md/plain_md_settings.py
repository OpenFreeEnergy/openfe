# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for plain MD Protocols using OpenMM + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.openmm_md.plain_md_methods.py`

"""
from openfe.protocols.openmm_utils.omm_settings import (
    Settings,
    SystemSettings,
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    SimulationSettingsMD,
    IntegratorSettings
)
from gufe.settings import SettingsBaseModel


class RepeatSettings(SettingsBaseModel):
    """Settings for how many independent MD runs to perform."""

    n_repeats: int = 1
    """
    Number of independent repeats to run.  Default 1
    """


class PlainMDProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    # Things for creating the systems
    system_settings: SystemSettings
    solvation_settings: OpenMMSolvationSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: SimulationSettingsMD

    # Setting number of repeats of md simulation
    repeat_settings: RepeatSettings
