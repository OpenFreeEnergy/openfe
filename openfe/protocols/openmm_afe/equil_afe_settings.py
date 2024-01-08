# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium AFE Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run absolute free
energies using OpenMM.

See Also
--------
openfe.protocols.openmm_afe.AbsoluteSolvationProtocol

TODO
----
* Add support for restraints

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

    Empty place holder for right now.
    """


class LambdaSettings(SettingsBaseModel):
    """Settings for lambda schedule
    """
    lambda_elec: list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    List of floats of lambda values for the electrostatics. 
    Zero means state A and 1 means state B.
    """
    lambda_vdw: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    """
    List of floats of lambda values for the van der Waals.
    """
    lambda_restraints: list = None
    """
    List of floats of lambda values for the restraints.
    """

    @validator('lambda_elec', 'lambda_vdw', 'lambda_restraints')
    def must_be_between_0_and_1(cls, v):
        for window in v:
            if not 0 <= window <= 1:
                errmsg = "Lambda windows must be between 0 and 1 "
                raise ValueError(errmsg)
        return v

    @validator('lambda_elec')
    def must_have_equal_number_elec_vdw_restraints_windows(cls, v, values):
        if values['lambda_restraints'] is None:
            if not len(v) == len(values['lambda_vdw']):
                errmsg = (
                    "Components elec and vdw must have equal amount of "
                    "lambda windows")
                raise ValueError(errmsg)
        else:
            if not len(v) == len(values['lambda_vdw']) == len(
                    values['lambda_restraints']):
                errmsg = (
                    "Components elec, vdw and restraints must have equal amount "
                    "of lambda windows")
                raise ValueError(errmsg)
        return v


class AbsoluteSolvationSettings(Settings):
    """
    Configuration object for ``AbsoluteSolvationProtocol``.

    See Also
    --------
    openfe.protocols.openmm_afe.AbsoluteSolvationProtocol
    """
    class Config:
        arbitrary_types_allowed = True

    # Inherited things
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields"""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters"""

    # Things for creating the systems
    vacuum_system_settings: SystemSettings
    """
    Simulation system settings including the
    long-range non-bonded methods for the vacuum transformation.
    """
    solvent_system_settings: SystemSettings
    """
    Simulation system settings including the
    long-range non-bonded methods for the solvent transformation.
    """
    solvation_settings: SolvationSettings
    """Settings for solvating the system."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings.
    """
    lambda_settings: LambdaSettings
    """
    Settings for controlling the lambda schedule for the different components 
    (vdw, elec, restraints).
    """
    alchemsampler_settings: AlchemicalSamplerSettings
    """
    Settings for controlling how we sample alchemical space, including the
    number of repeats.
    """

    # MD Engine things
    vacuum_engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform
    for the vacuum transformation.
    """
    solvent_engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform
    for the solvent transformation.
    """

    # Sampling State defining things
    integrator_settings: IntegratorSettings
    """
    Settings for controlling the integrator, such as the timestep and
    barostat settings.
    """

    # Simulation run settings
    vacuum_simulation_settings: SimulationSettings
    """
    Simulation control settings, including simulation lengths and
    record-keeping for the vacuum transformation.
    """
    solvent_simulation_settings: SimulationSettings
    """
    Simulation control settings, including simulation lengths and
    record-keeping for the solvent transformation.
    """
