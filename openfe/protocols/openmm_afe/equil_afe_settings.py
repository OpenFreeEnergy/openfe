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
from gufe.settings import (
    Settings,
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    SystemSettings,
    SolvationSettings,
    AlchemicalSamplerSettings,
    OpenMMEngineSettings,
    IntegratorSettings,
    SimulationSettings
)


try:
    from pydantic.v1 import validator
except ImportError:
    from pydantic import validator  # type: ignore[assignment]


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

    # Inherited things
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields"""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters"""

    # Things for creating the systems
    system_settings: SystemSettings
    """
    Simulation system settings including the
    long-range non-bonded methods.
    """
    solvation_settings: SolvationSettings
    """Settings for solvating the system."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings including lambda windows.
    """
    alchemsampler_settings: AlchemicalSamplerSettings
    """
    Settings for controling how we sample alchemical space, including the
    number of repeats.
    """

    # MD Engine things
    engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform.
    """

    # Sampling State defining things
    integrator_settings: IntegratorSettings
    """
    Settings for controlling the integrator, such as the timestep and
    barostat settings.
    """

    # Simulation run settings
    simulation_settings: SimulationSettings
    """
    Simulation control settings, including simulation lengths and
    record-keeping.
    """
