# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for plain MD Protocols using OpenMM + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.openmm_md.plain_md_methods.py`

"""
from gufe.settings import SettingsBaseModel

from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    MDOutputSettings,
    MDSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    Settings,
)

try:
    from pydantic.v1 import validator
except ImportError:
    from pydantic import validator  # type: ignore[assignment]


class PlainMDProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    protocol_repeats: int
    """
    Number of independent MD runs to perform.
    """

    @validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # Things for creating the systems
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
