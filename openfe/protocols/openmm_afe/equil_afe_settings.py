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
    SimulationSettings,
    OutputSettings,
)
import numpy as np


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
    lambda_elec: list[float] = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]
    """
    List of floats of lambda values for the electrostatics. 
    Zero means state A and 1 means state B.
    Length of this list needs to match length of lambda_vdw and lambda_restraints.
    """
    lambda_vdw: list[float] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
    ]
    """
    List of floats of lambda values for the van der Waals.
    Zero means state A and 1 means state B.
    Length of this list needs to match length of lambda_elec and lambda_restraints.
    """
    lambda_restraints: list[float] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    """
    List of floats of lambda values for the restraints.
    Zero means state A and 1 means state B.
    Length of this list needs to match length of lambda_vdw and lambda_elec.
    """

    @validator('lambda_elec', 'lambda_vdw', 'lambda_restraints')
    def must_be_between_0_and_1(cls, v):
        for window in v:
            if not 0 <= window <= 1:
                errmsg = ("Lambda windows must be between 0 and 1, got a"
                          f" window with value {window}.")
                raise ValueError(errmsg)
        return v

    @validator('lambda_elec', 'lambda_vdw', 'lambda_restraints')
    def must_be_monotonic(cls, v):

        difference = np.diff(v)

        if not all(i >= 0. for i in difference):
            errmsg = f"The lambda schedule is not monotonic, got schedule {v}."
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

    protocol_repeats: int = 3
    """
    The number of completely independent repeats of the entire sampling 
    process. The mean of the repeats defines the final estimate of FE 
    difference, while the variance between repeats is used as the uncertainty.  
    Default 3
    """

    @validator('protocol_repeats')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

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
    Settings for controlling how we sample alchemical space.
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
    Simulation control settings, including simulation lengths
    for the vacuum transformation.
    """
    solvent_simulation_settings: SimulationSettings
    """
    Simulation control settings, including simulation lengths
    for the solvent transformation.
    """
    vacuum_output_settings: OutputSettings
    """
    Simulation output settings for the vacuum transformation.
    """
    solvent_output_settings: OutputSettings
    """
    Simulation output settings for the solvent transformation.
    """
