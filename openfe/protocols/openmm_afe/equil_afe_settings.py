# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using :class:`openfe.protocols.openmm_abfe.equil_abfe_methods.py`


TODO
----
* Add support for restraints
* Improve this docstring by adding an example use case.

"""
from typing import Optional
from pydantic import validator
from openff.units import unit
from openfe.protocols.openmm_utils.omm_settings import (
    Settings, SettingsBaseModel, ThermoSettings,
    OpenMMSystemGeneratorFFSettings, SystemSettings,
    SolvationSettings, AlchemicalSamplerSettings,
    OpenMMEngineSettings, IntegratorSettings,
    SimulationSettings
)



class AlchemicalSettings(SettingsBaseModel):
    """Settings for the alchemical protocol

    These settings describe the lambda schedule and the creation of the
    hybrid system.
    """

    lambda_elec_windows = 12
    """Number of lambda electrostatic alchemical steps, default 12"""
    lambda_vdw_windows = 12
    """Number of lambda vdw alchemical steps, default 12"""

    @validator('lambda_elec_windows', 'lambda_vdw_windows')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = ("Number of lambda steps must be positive ")
            raise ValueError(errmsg)
        return v


class AbsoluteTransformSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    # Things for creating the systems
    system_settings: SystemSettings
    solvation_settings: SolvationSettings

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    alchemsampler_settings: AlchemicalSamplerSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings: SimulationSettings
