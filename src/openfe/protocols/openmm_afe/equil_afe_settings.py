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

import numpy as np
from gufe.settings import (
    OpenMMSystemGeneratorFFSettings,
    SettingsBaseModel,
    ThermoSettings,
)
from pydantic import field_validator

from openfe.protocols.openmm_utils.omm_settings import (
    BaseSolvationSettings,
    IntegratorSettings,
    MDOutputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)
from openfe.protocols.restraint_utils.settings import (
    BaseRestraintSettings,
    BoreschRestraintSettings,
)


class AlchemicalSettings(SettingsBaseModel):
    """
    Alchemical settings for Protocols which use the
    AbsoluteAlchemicalFactory.
    """

    disable_alchemical_dispersion_correction: bool = False
    """
    If True, the long-range dispersion correction will not
    be included for the alchemical region, avoiding the need
    to recompute the correction. This can improve performance,
    at the cost of accuracy. Default is False.
    """
    annihilate_sterics: bool = False
    """
    If True, sterics (Lennard-Jones) will be annhilated instead
    of decoupled. Default is False.
    """
    softcore_alpha: float = 0.5
    """
    Alchemical softcore parameter for the Lennard-Jones interactions
    (default is 0.5).

    The generalized softcore potential formalism introduced by
    Pham and Shirts, J. Chem. Phys. 135, 034114 (2011), equation 13,
    is used here. The ``softcore_a``, ``softcore_b``, and
    ``softcore_c`` parameters are used alongside ``softcore_alpha``
    to control how the potential is scaled.
    """
    softcore_a: float = 1.0
    """
    Scaling constant ``a`` in
    Eq. 13 from Pham and Shirts, J. Chem. Phys. 135, 034114 (2011).
    """
    softcore_b: float = 1.0
    """
    Scaling constant ``b`` in
    Eq. 13 from Pham and Shirts, J. Chem. Phys. 135, 034114 (2011).
    """
    softcore_c: float = 6.0
    """
    Scaling constant ``c`` in
    Eq. 13 from Pham and Shirts, J. Chem. Phys. 135, 034114 (2011).
    """


class LambdaSettings(SettingsBaseModel):
    """Lambda schedule settings.

    Defines lists of floats to control various aspects of the alchemical
    transformation.

    Notes
    -----
    * In all cases a lambda value of 0 defines the system in state A, whilst
      a value of 1 defines the system in state B. In an absolute transformation,
      state A means a fully interacting ligand without any restraints applied,
      and state B means a fully non-interacting ligand, with optional restraints
      applied.
    * ``lambda_elec``, ``lambda_vdw``, and ``lambda_restraints`` must all be of
      the same length, defining all the windows of the transformation.
    """

    # fmt: off
    lambda_elec: list[float] = [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]
    # fmt: on
    """
    List of floats of lambda values for the electrostatics.
    Zero means fully interacting (state A),
    and one means annihilated (state B).
    Length of this list needs to match length of lambda_vdw and
    lambda_restraints.
    """
    # fmt: off
    lambda_vdw: list[float] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
    ]
    # fmt: on
    """
    List of floats of lambda values for the van der Waals.
    Zero means full interacting (state A) and one means decoupled (state B).
    Length of this list needs to match length of lambda_elec and
    lambda_restraints.
    """
    # fmt: off
    lambda_restraints: list[float] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    # fmt: on
    """
    List of floats of lambda values for the restraints.
    Zero means no restraints are applied (state A), and
    one means restraints are fully applied (state B).

    Note: The length of this list needs to match length of lambda_vdw and lambda_elec.
    """

    @field_validator("lambda_elec", "lambda_vdw", "lambda_restraints")
    def must_be_between_0_and_1(cls, v):
        for window in v:
            if not 0 <= window <= 1:
                errmsg = (
                    f"Lambda windows must be between 0 and 1, got a window with value {window}."
                )
                raise ValueError(errmsg)
        return v

    @field_validator("lambda_elec", "lambda_vdw", "lambda_restraints")
    def must_be_monotonic(cls, v):
        difference = np.diff(v)
        monotonic = np.all(difference >= 0)

        if not monotonic:
            errmsg = (
                "The lambda schedule is not monotonically increasing, "
                f"got the following schedule: {v}."
            )
            raise ValueError(errmsg)

        return v


class ABFEPreEquilOutputSettings(MDOutputSettings):
    output_indices: str = "all"
    """
    Selection string for which part of the system to write coordinates for.
    For now, must be "all".
    """

    equil_nvt_structure: str | None = "equil_nvt_structure.pdb"
    """
    Name of the PDB file containing the system after NVT pre-equilibration.
    Only the atom subset specified by output_indices is saved.
    Default 'equil_nvt_structure.pdb'.
    """

    equil_npt_structure: str | None = "equil_npt_structure.pdb"
    """
    Name of the PDB file containing the system after NPT pre-equilibration.
    Only the atom subset specified by output_indices is saved.
    Default 'equil_npt_structure.pdb'.
    """

    production_trajectory_filename: str | None = "production_equil.xtc"
    """
    Name pre-equilibration "production" (i.e. extended NPT) trajectory file.
    Only the atom subset specified by output_indices is saved.
    Default `production_equil.xtc`.
    """

    log_output: str | None = "production_equil_simulation.log"
    """
    Filename for writing the pre-equilibration extended NPT MD simulation
    log file. This includes ns/day, timesteps, energies, density, etc.
    Default 'production_equil_simulation.log'
    """

    @field_validator("output_indices")
    def must_be_all(cls, v):
        # Would be better if this was just changed to a Literal
        # but changing types in child classes in pydantic is messy
        if v != "all":
            msg = "output_indices must be all for ABFE pre-equilibration simulations"
            raise ValueError(msg)
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

    @field_validator("protocol_repeats")
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
    vacuum_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the vacuum transformation.
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


class AbsoluteBindingSettings(SettingsBaseModel):
    """
    Configuration object for ``AbsoluteBindingPProtocol``

    See Also
    --------
    openfe.protocols.openmm_afe.AbsoluteBindingProtocol
    """

    protocol_repeats: int
    """
    The number of completely independent repeats of the entire sampling
    process. The mean of the repeats defines the final estimate of FE
    difference, while the variance between repeats is used as the uncertainty.
    """

    @field_validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    forcefield_settings: OpenMMSystemGeneratorFFSettings
    """Parameters to set up the force field with OpenMM Force Fields"""
    thermo_settings: ThermoSettings
    """Settings for thermodynamic parameters"""

    solvent_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the system in the solvent."""
    complex_solvation_settings: OpenMMSolvationSettings
    """Settings for solvating the system in the complex."""

    # Alchemical settings
    alchemical_settings: AlchemicalSettings
    """
    Alchemical protocol settings.
    """
    complex_lambda_settings: LambdaSettings
    """
    Settings for controlling the complex transformation leg
    lambda schedule for the different components (vdw, elec, restraints).
    """
    solvent_lambda_settings: LambdaSettings
    """
    Settings for controlling the solvent transformation leg
    lambda schedule for the different components (vdw, elec, restraints).

    Notes
    -----
    * The `restraints` entry of the lambda settings will be ignored in the
      solvent leg.
    """

    # MD Engine things
    engine_settings: OpenMMEngineSettings
    """
    Settings specific to the OpenMM engine, such as the compute platform.
    """

    # Sampling State defining things
    solvent_integrator_settings: IntegratorSettings
    """
    Settings for controlling the integrator, such as the timestep and
    barostat settings in the solvent.
    """
    complex_integrator_settings: IntegratorSettings
    """
    Settings for controlling the integrator, such as the timestep and
    barostat settings in the complex.
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

    # Simulation output settings
    complex_equil_output_settings: ABFEPreEquilOutputSettings
    """
    Simulation output settings for the complex non-alchemical equilibration.
    """
    complex_output_settings: MultiStateOutputSettings
    """
    Simulation output settings for the complex transformation.
    """
    solvent_equil_output_settings: ABFEPreEquilOutputSettings
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
    restraint_settings: BaseRestraintSettings
    """
    Settings controlling how restraints are added to the system in the
    complex simulation.
    """
