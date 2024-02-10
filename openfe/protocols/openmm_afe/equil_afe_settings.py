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
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    MultiStateSimulationSettings,
    BaseSolvationSettings,
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    IntegratorSettings,
    OutputSettings,
    MDSimulationSettings,
    MDOutputSettings,
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
    """Lambda schedule settings.

    Defines lists of floats to control various aspects of the alchemical
    transformation.

    Notes
    -----
    * In all cases a lambda value of 0 defines a fully interacting state A and
      a non-interacting state B, whilst a value of 1 defines a fully interacting
      state B and a non-interacting state A.
    * ``lambda_elec``, `lambda_vdw``, and ``lambda_restraints`` must all be of
      the same length, defining all the windows of the transformation.

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


# This subclasses from SettingsBaseModel as it has vacuum_forcefield and
# solvent_forcefield fields, not just a single forcefield_settings field
class AbsoluteSolvationSettings(SettingsBaseModel):
    """
    Configuration object for ``AbsoluteSolvationProtocol``.

    See Also
    --------
    openfe.protocols.openmm_afe.AbsoluteSolvationProtocol
    """
    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling 
    process. The mean of the repeats defines the final estimate of FE 
    difference, while the variance between repeats is used as the uncertainty.  
    """

    @validator('protocol_repeats')
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # Inherited things
    solvent_forcefield_settings: OpenMMSystemGeneratorFFSettings
    vacuum_forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields"""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters"""

    solvation_settings: OpenMMSolvationSettings
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
    vacuum_equil_simulation_settings: MDSimulationSettings
    """
    Pre-alchemical vacuum simulation control settings.

    Notes
    -----
    The `NVT` equilibration should be set to 0 * unit.nanosecond
    as it will not be run.
    """
    vacuum_simulation_settings: MultiStateSimulationSettings
    """
    Simulation control settings, including simulation lengths
    for the vacuum transformation.
    """
    solvent_equil_simulation_settings: MDSimulationSettings
    """
    Pre-alchemical solvent simulation control settings.
    """
    solvent_simulation_settings: MultiStateSimulationSettings
    """
    Simulation control settings, including simulation lengths
    for the solvent transformation.
    """
    vacuum_equil_output_settings: MDOutputSettings
    """
    Simulation output settings for the vacuum non-alchemical equilibration.
    """
    vacuum_output_settings: OutputSettings
    """
    Simulation output settings for the vacuum transformation.
    """
    solvent_equil_output_settings: MDOutputSettings
    """
    Simulation output settings for the solvent non-alchemical equilibration.
    """
    solvent_output_settings: OutputSettings
    """
    Simulation output settings for the solvent transformation.
    """
