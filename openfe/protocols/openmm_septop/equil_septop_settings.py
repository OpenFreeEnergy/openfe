# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Settings class for equilibrium SepTop Protocols using OpenMM + OpenMMTools

This module implements the necessary settings necessary to run SepTop RBFE
calculations using OpenMM.

See Also
--------
openfe.protocols.openmm_septop.SepTopProtocol
"""
from typing import Optional

import numpy as np
from gufe.settings import (
    OpenMMSystemGeneratorFFSettings,
    SettingsBaseModel,
    ThermoSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    MDOutputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)
from openfe.protocols.restraint_utils.settings import BaseRestraintSettings
from gufe.vendor.openff.models.types import FloatQuantity
from openff.units import unit
from pydantic.v1 import validator


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

    lambda_elec_A: list[float] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    """
    List of floats of the lambda values for the electrostatics of ligand A.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and lambda_restraints.
    """
    lambda_elec_B: list[float] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.75,
        0.5,
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    """
    List of floats of the lambda values for the electrostatics of ligand B.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and
    lambda_restraints.
    """
    lambda_vdw_A: list[float] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857142,
        0.8571428571428571,
        1.0,
    ]
    """
    List of floats of lambda values for the van der Waals of ligand A.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_elec and
    lambda_restraints.
    """
    lambda_vdw_B: list[float] = [
        1.0,
        0.8571428571428572,
        0.7142857142857143,
        0.5714285714285714,
        0.4285714285714286,
        0.2857142857142858,
        0.1428571428571429,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    """
    List of floats of lambda values for the van der Waals of ligand B.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_elec and lambda_restraints.
    """
    lambda_restraints_A: list[float] = [
        0.0,
        0.05,
        0.1,
        0.3,
        0.5,
        0.75,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    """
    List of floats of lambda values for the restraints of ligand A.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and lambda_elec.
    """
    lambda_restraints_B: list[float] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.75,
        0.5,
        0.3,
        0.1,
        0.05,
        0.0,
    ]
    """
    List of floats of lambda values for the restraints of ligand B.
    Zero means fully interacting and 1 means fully decoupled.
    Length of this list needs to match length of lambda_vdw and lambda_elec.
    """

    @validator(
        "lambda_elec_A",
        "lambda_elec_B",
        "lambda_vdw_A",
        "lambda_vdw_B",
        "lambda_restraints_A",
        "lambda_restraints_B",
    )
    def must_be_between_0_and_1(cls, v):
        for window in v:
            if not 0 <= window <= 1:
                errmsg = (
                    "Lambda windows must be between 0 and 1, got a"
                    f" window with value {window}."
                )
                raise ValueError(errmsg)
        return v

    @validator(
        "lambda_elec_A",
        "lambda_elec_B",
        "lambda_vdw_A",
        "lambda_vdw_B",
        "lambda_restraints_A",
        "lambda_restraints_B",
    )
    def must_be_monotonic(cls, v):

        difference = np.diff(v)

        monotonic = np.all(difference <= 0) or np.all(difference >= 0)

        if not monotonic:
            errmsg = f"The lambda schedule is not monotonic, got schedule {v}."
            raise ValueError(errmsg)

        return v


class SepTopEquilOutputSettings(MDOutputSettings):
    # reporter settings
    output_indices = "all"
    production_trajectory_filename: Optional[str] = "simulation"
    """
    Basename for the path to the storage file for analysis. The protocol will
    append a '_stateA.xtc' and a '_stateB.xtc' for the output files of the
    respective endstates. Default 'simulation'.
    """
    trajectory_write_interval: FloatQuantity["picosecond"] = 20.0 * unit.picosecond
    """
    Frequency to write the xtc file. Default 20 * unit.picosecond.
    """
    preminimized_structure: Optional[str] = "system"
    """
    Basename for the path to the pdb file of the full pre-minimized systems.
    The protocol will append a '_stateA.pdb' and a '_stateB.pdb' for the output
    files of the respective endstates. Default 'system'.
    """
    minimized_structure: Optional[str] = "minimized"
    """
    Basename for the path to the pdb file of the systems after minimization.
    The protocol will append a '_stateA.pdb' and a '_stateB.pdb' for the output
    files of the respective endstates. Default 'minimized'.
    """
    equil_nvt_structure: Optional[str] = "equil_nvt"
    """
    Basename for the path to the pdb file of the systems after NVT equilibration.
    The protocol will append a '_stateA' and a '_stateB' for the output files
    of the respective endstates. Default 'equil_nvt.pdb'.
    """
    equil_npt_structure: Optional[str] = "equil_npt"
    """
    Basename for the path to the pdb file of the systems after NPT equilibration.
    The protocol will append a '_stateA.pdb' and a '_stateB.pdb' for the output
    files of the respective endstates. Default 'equil_npt'.
    """
    log_output: Optional[str] = "simulation"
    """
    Basename for the filename for writing the log of the MD simulation,
    including timesteps, energies, density, etc.
    The protocol will append a '_stateA.pdb' and a '_stateB.pdb' for the output
    files of the respective endstates. Default 'simulation'.
    """


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

    @validator("protocol_repeats")
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

    solvent_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the solvent system."""

    complex_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the complex system."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings.
    """
    solvent_lambda_settings: LambdaSettings
    """
    Settings for controlling the lambda schedule for the different components
    (vdw, elec, restraints) in the solvent.
    """
    complex_lambda_settings: LambdaSettings
    """
    Settings for controlling the lambda schedule for the different components
    (vdw, elec, restraints) in the complex.
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
    complex_equil_output_settings: SepTopEquilOutputSettings
    """
    Simulation output settings for the complex non-alchemical equilibration.
    """
    complex_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the complex transformation.
    """
    solvent_equil_output_settings: SepTopEquilOutputSettings
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
    solvent_restraint_settings: BaseRestraintSettings
    """
    Settings for the harmonic restraint in the solvent
    """
    complex_restraint_settings: BaseRestraintSettings
    """
    Settings for the Boresch restraints in the complex
    """
