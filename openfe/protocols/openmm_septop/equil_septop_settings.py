# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium SepTop Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run SepTop RBFE
calculations using OpenMM.

See Also
--------
openfe.protocols.openmm_septop.SepTopProtocol
"""
from gufe.settings import (
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    MultiStateSimulationSettings,
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    IntegratorSettings,
    OpenFFPartialChargeSettings,
    MultiStateOutputSettings,
    MDSimulationSettings,
    MDOutputSettings,
)
from openff.units import unit
from openff.models.types import FloatQuantity
import numpy as np
from pydantic.v1 import validator


class AlchemicalSettings(SettingsBaseModel):
    """Settings for the alchemical protocol

    Empty place holder for right now.
    """


class RestraintsSettings(SettingsBaseModel):
    """
    Settings for the restraints.
    """
    k_distance: FloatQuantity['kJ/(mol*nanometers**2)'] = 1000 * unit.kilojoule_per_mole / unit.nanometer**2


class SolventRestraintsSettings(RestraintsSettings):
    """
    Settings for the harmonic restraint in the solvent
    """


class ComplexRestraintsSettings(RestraintsSettings):
    """
    Settings for the Boresch restraints in the complex
    """
    class Config:
        arbitrary_types_allowed = True

    k_theta: FloatQuantity['kJ/(mol*rad**2)'] = 83.68 * unit.kilojoule_per_mole / unit.radians ** 2


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
    lambda_elec_ligandA: list[float] = [0.0] * 8 + [0.25, 0.5, 0.75] + [1.0] * 8
    """
    List of floats of the lambda values for the electrostatics of ligand A. 
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and lambda_restraints.
    """
    lambda_elec_ligandB: list[float] = [1.0] * 8 + [0.75, 0.5, 0.25] + [0.0] * 8
    """
    List of floats of the lambda values for the electrostatics of ligand B. 
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and 
    lambda_restraints.
    """
    lambda_vdw_ligandA: list[float] = [0.0] * 8 + [
        0.00, 0.0, 0.00] + np.linspace(0.0, 1.0, 8).tolist()
    """
    List of floats of lambda values for the van der Waals of ligand A.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_elec and 
    lambda_restraints.
    """
    lambda_vdw_ligandB: list[float] = np.linspace(1.0, 0.0, 8).tolist() + [
        0.0, 0.0, 0.0] + [0.0] * 8
    """
    List of floats of lambda values for the van der Waals of ligand B.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_elec and lambda_restraints.
    """
    lambda_restraints_ligandA: list[float] = [
                                                 0.0, 0.05, 0.1, 0.3, 0.5,
                                                 0.75, 1.0, 1.0] + [
                                                 1.0] * 3 + [1.0] * 8
    """
    List of floats of lambda values for the restraints of ligand A.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and lambda_elec.
    """
    lambda_restraints_ligandB: list[float] = [1.0] * 8 + [1.0] * 3 + [
        1.0, 0.95, 0.9, 0.7, 0.5, 0.25, 0.0, 0.0]
    """
    List of floats of lambda values for the restraints of ligand B.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and lambda_elec.
    """


    @validator('lambda_elec_ligandA', 'lambda_elec_ligandB',
               'lambda_vdw_ligandA', 'lambda_vdw_ligandB',
               'lambda_restraints_ligandA', 'lambda_restraints_ligandB')
    def must_be_between_0_and_1(cls, v):
        for window in v:
            if not 0 <= window <= 1:
                errmsg = ("Lambda windows must be between 0 and 1, got a"
                          f" window with value {window}.")
                raise ValueError(errmsg)
        return v

    @validator('lambda_elec_ligandA', 'lambda_elec_ligandB',
               'lambda_vdw_ligandA', 'lambda_vdw_ligandB',
               'lambda_restraints_ligandA', 'lambda_restraints_ligandB')
    def must_be_monotonic(cls, v):

        difference = np.diff(v)

        monotonic = np.all(difference <= 0) or np.all(difference >= 0)

        if not monotonic:
            errmsg = f"The lambda schedule is not monotonic, got schedule {v}."
            raise ValueError(errmsg)

        return v


# This subclasses from SettingsBaseModel as it has vacuum_forcefield and
# solvent_forcefield fields, not just a single forcefield_settings field
class SepTopSettings(SettingsBaseModel):
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
    complex_forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields"""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters"""

    solvent_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the solvent system."""

    complex_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the complex system."""

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
    complex_engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform
    for the complex transformation.
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
    complex_equil_simulation_settings: MDSimulationSettings
    """
    Pre-alchemical complex simulation control settings.
    """
    complex_simulation_settings: MultiStateSimulationSettings
    """
    Simulation control settings, including simulation lengths
    for the complex transformation.
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
    complex_equil_output_settings: MDOutputSettings
    """
    Simulation output settings for the complex non-alchemical equilibration.
    """
    complex_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the complex transformation.
    """
    solvent_equil_output_settings: MDOutputSettings
    """
    Simulation output settings for the solvent non-alchemical equilibration.
    """
    solvent_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the solvent transformation.
    """
    partial_charge_settings: OpenFFPartialChargeSettings
    """
    Settings for controlling how to assign partial charges,
    including the partial charge assignment method, and the
    number of conformers used to generate the partial charges.
    """
    solvent_restraints_settings: SolventRestraintsSettings
    """
    Settings for the harmonic restraint in the solvent
    """
    complex_restraints_settings: ComplexRestraintsSettings
    """
    Settings for the Boresch restraints in the complex
    """
