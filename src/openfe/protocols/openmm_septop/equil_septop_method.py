# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium SepTop RBFE Protocol --- :mod:`openfe.protocols.openmm_septop.equil_septop_method`
========================================================================================================

This module implements the necessary methodology tooling to run a
Separated Topologies RBFE calculation using OpenMM tools and one of the
following alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------

* Transformations that involve net charge changes are currently not supported.
  The ligands must have the same net charge.
* Only small molecules are allowed to act as alchemical molecules.
  Alchemically changing protein or solvent components would induce
  perturbations which are too large to be handled by this Protocol.


Acknowledgements
----------------
This Protocol is based on and inspired by the SepTop implementation from
the Mobleylab (https://github.com/MobleyLab/SeparatedTopologies) as well as
femto (https://github.com/Psivant/femto).

"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections import defaultdict
from typing import Any, Iterable, Optional, Union

import gufe
from gufe import (
    ChemicalSystem,
    ProteinComponent,
    ProteinMembraneComponent,
    SmallMoleculeComponent,
    SolvatedPDBComponent,
    SolventComponent,
    settings,
)
from gufe.components import Component
from openff.units import unit as offunit
from rdkit import Chem

from openfe.due import Doi, due
from openfe.protocols.openmm_septop.equil_septop_settings import (
    AlchemicalSettings,
    IntegratorSettings,
    LambdaSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    SepTopEquilOutputSettings,
    SepTopSettings,
    SettingsBaseModel,
)

from ..openmm_utils import settings_validation, system_validation
from ..restraint_utils.settings import (
    BoreschRestraintSettings,
    DistanceRestraintSettings,
)
from .septop_protocol_results import SepTopProtocolResult
from .septop_units import (
    SepTopComplexAnalysisUnit,
    SepTopComplexRunUnit,
    SepTopComplexSetupUnit,
    SepTopSolventAnalysisUnit,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)

due.cite(
    Doi("10.1021/acs.jctc.3c00282"),
    description="Separated Topologies method",
    path="openfe.protocols.openmm_septop.equil_septop_method",
    cite_module=True,
)

due.cite(
    Doi("10.5281/zenodo.596622"),
    description="OpenMMTools",
    path="openfe.protocols.openmm_septop.equil_septop_method",
    cite_module=True,
)

due.cite(
    Doi("10.1371/journal.pcbi.1005659"),
    description="OpenMM",
    path="openfe.protocols.openmm_septop.equil_septop_method",
    cite_module=True,
)


logger = logging.getLogger(__name__)


def _check_alchemical_charge_difference(
    ligandA: SmallMoleculeComponent,
    ligandB: SmallMoleculeComponent,
):
    """
    Checks and returns the difference in formal charge between state A
    and B.

    Raises
    ------
    ValueError
      * If a change in net charge is detected.

    Parameters
    ----------
    ligandA: SmallMoleculeComponent
    ligandB: SmallMoleculeComponent
    """
    chg_A = Chem.rdmolops.GetFormalCharge(ligandA.to_rdkit())
    chg_B = Chem.rdmolops.GetFormalCharge(ligandB.to_rdkit())

    difference = chg_A - chg_B

    if abs(difference) != 0:
        errmsg = (
            f"A charge difference of {difference} is observed "
            "between the end states. Unfortunately this protocol "
            "currently does not support net charge changes."
        )
        raise ValueError(errmsg)


class SepTopProtocol(gufe.Protocol):
    """
    SepTop RBFE calculations using OpenMM and OpenMMTools.

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_septop.SepTopSettings`
    :class:`openfe.protocols.openmm_septop.SepTopProtocolResult`
    :class:`openfe.protocols.openmm_septop.SepTopComplexSetupUnit`
    :class:`openfe.protocols.openmm_septop.SepTopComplexRunUnit`
    :class:`openfe.protocols.openmm_septop.SepTopSolventSetupUnit`
    :class:`openfe.protocols.openmm_septop.SepTopSolventRunUnit`
    """

    result_cls = SepTopProtocolResult
    _settings_cls = SepTopSettings
    _settings: SepTopSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        Settings
          a set of default settings
        """
        return SepTopSettings(
            protocol_repeats=3,
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
            solvent_lambda_settings=LambdaSettings(
                lambda_elec_A=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                ],
                lambda_elec_B=[
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                lambda_vdw_A=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.15, 0.23, 0.3, 0.4, 0.52, 0.64, 0.76, 0.88, 1.0,
                ],
                lambda_vdw_B=[
                    1.0, 0.85, 0.77, 0.7, 0.6, 0.48, 0.36, 0.24, 0.12,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                lambda_restraints_A=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                lambda_restraints_B=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            ),
            complex_lambda_settings=LambdaSettings(),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvent_solvation_settings=OpenMMSolvationSettings(),
            complex_solvation_settings=OpenMMSolvationSettings(
                solvent_padding=1.0 * offunit.nanometer,
            ),
            engine_settings=OpenMMEngineSettings(),
            solvent_integrator_settings=IntegratorSettings(),
            complex_integrator_settings=IntegratorSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.1 * offunit.nanosecond,
                production_length=2.0 * offunit.nanosecond,
            ),
            solvent_equil_output_settings=SepTopEquilOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure="equil_npt",
                production_trajectory_filename="equil_npt",
                log_output="equil_simulation",
            ),
            solvent_simulation_settings=MultiStateSimulationSettings(
                n_replicas=27,
                minimization_steps=5000,
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename="solvent.nc",
                checkpoint_storage_filename="solvent_checkpoint.nc",
            ),
            complex_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.1 * offunit.nanosecond,
                production_length=2.0 * offunit.nanosecond,
            ),
            complex_equil_output_settings=SepTopEquilOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure="equil_npt",
                production_trajectory_filename="equil_production",
                log_output="equil_simulation",
            ),
            complex_simulation_settings=MultiStateSimulationSettings(
                n_replicas=19,
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            complex_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename="complex.nc",
                checkpoint_storage_filename="complex_checkpoint.nc",
            ),
            solvent_restraint_settings=DistanceRestraintSettings(
                spring_constant=1000.0 * offunit.kilojoule_per_mole / offunit.nanometer**2,
            ),
            complex_restraint_settings=BoreschRestraintSettings(),
        )  # fmt: skip

    @classmethod
    def _adaptive_settings(
        cls,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        initial_settings: None | SepTopSettings = None,
    ) -> SepTopSettings:
        """
        Get the recommended OpenFE settings for this Protocol based on the input states involved in the
        transformation.

        These are intended as a suitable starting point, which can be further
        customized before creating a Protocol.

        Parameters
        ----------
        stateA : ChemicalSystem
            The initial state of the transformation.
        stateB : ChemicalSystem
            The final state of the transformation.
        initial_settings : None | SepTopSettings, optional
            Initial settings to adapt. If None, default settings are used.

        Returns
        -------
        SepTopSettings
            The recommended settings for this protocol based on the input states.
        """
        # use initial settings or default settings
        if initial_settings is not None:
            protocol_settings = initial_settings.model_copy(deep=True)
        else:
            protocol_settings = cls.default_settings()

        # adapt the barostat based on the ProteinComponent
        if stateA.contains(ProteinMembraneComponent):
            protocol_settings.complex_integrator_settings.barostat = "MonteCarloMembraneBarostat"

        return protocol_settings

    @staticmethod
    def _validate_endstates(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
    ) -> None:
        """
        A complex relative transformation is defined (in terms of gufe components)
        as starting from one or more ligands and a protein in solvent and
        ending up in a state with one ligand that is different.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If there is no SolventComponent and no ProteinComponent
          in either stateA or stateB.
          If there are no or more than one alchemical components in state A.
          If there are no or more than one alchemical components in state B.
          If there are any alchemical components that are not SmallMoleculeComponents.
          If a change in net charge between the alchemical components is detected.
        """
        # check that there is a protein component
        if not stateA.contains(ProteinComponent):
            errmsg = "No ProteinComponent found in stateA"
            raise ValueError(errmsg)

        if not stateB.contains(ProteinComponent):
            errmsg = "No ProteinComponent found in stateB"
            raise ValueError(errmsg)

        # check that there is only one protein component
        system_validation.validate_protein(stateA)
        system_validation.validate_protein(stateB)

        # check that there is a SolventComponent
        if not stateA.contains(SolventComponent):
            errmsg = "No SolventComponent found in stateA"
            raise ValueError(errmsg)

        if not stateB.contains(SolventComponent):
            errmsg = "No SolventComponent found in stateB"
            raise ValueError(errmsg)

        # Check the difference between the endstates
        diff = stateA.component_diff(stateB)

        for i, state in enumerate(["stateA", "stateB"]):
            # Error if there isn't exactly one alchemical component
            if len(diff[i]) != 1:
                errmsg = (
                    "Only one alchemical species is supported. "
                    f"Number of unique components found in {state}: {len(diff[i])}."
                )
                raise ValueError(errmsg)

            # Error if the component isn't an SMC
            if not isinstance(diff[i][0], SmallMoleculeComponent):
                errmsg = (
                    "Only transforming SmallMoleculeComponents are supported "
                    f"by this Protocol. Found a {type(diff[i][0])}."
                )
                raise ValueError(errmsg)

        # Raise an error if there is a change in net charge
        _check_alchemical_charge_difference(diff[0][0], diff[1][0])

    @staticmethod
    def _validate_lambda_schedule(
        lambda_settings: LambdaSettings,
        simulation_settings: MultiStateSimulationSettings,
    ) -> None:
        """
        Checks that the lambda schedule is set up correctly.

        Parameters
        ----------
        lambda_settings : LambdaSettings
          the lambda schedule Settings
        simulation_settings : MultiStateSimulationSettings
          the settings for either the complex or solvent phase

        Raises
        ------
        ValueError
          If the number of lambda windows differs for electrostatics and sterics.
          If the number of replicas does not match the number of lambda windows.

        TODO
        ----
        Add a warning if all the lambda restraints are zero? Issue #1945.
        """

        lambda_elec_A = lambda_settings.lambda_elec_A
        lambda_elec_B = lambda_settings.lambda_elec_B
        lambda_vdw_A = lambda_settings.lambda_vdw_A
        lambda_vdw_B = lambda_settings.lambda_vdw_B
        lambda_restraints_A = lambda_settings.lambda_restraints_A
        lambda_restraints_B = lambda_settings.lambda_restraints_B
        n_replicas = simulation_settings.n_replicas

        # Ensure that all lambda components have equal amount of windows
        lambda_components = [
            lambda_vdw_A,
            lambda_vdw_B,
            lambda_elec_A,
            lambda_elec_B,
            lambda_restraints_A,
            lambda_restraints_B,
        ]
        lengths = {len(lam) for lam in lambda_components}
        if len(lengths) != 1:
            errmsg = (
                "Components elec, vdw, and restraints must have equal amount"
                f" of lambda windows. Got {len(lambda_elec_A)} and "
                f"{len(lambda_elec_B)} elec lambda windows, "
                f"{len(lambda_vdw_A)} and {len(lambda_vdw_B)} vdw "
                f"lambda windows, and {len(lambda_restraints_A)} and "
                f"{len(lambda_restraints_B)} restraints lambda windows."
            )
            raise ValueError(errmsg)

        # Ensure that number of overall lambda windows matches number of lambda
        # windows for individual components
        if n_replicas != len(lambda_vdw_B):
            errmsg = (
                f"Number of replicas {n_replicas} does not equal the"
                f" number of lambda windows {len(lambda_vdw_B)}"
            )
            raise ValueError(errmsg)

        # Check if there are lambda windows with naked charges
        for state, elec, vdw in (
            ("A", lambda_elec_A, lambda_vdw_A),
            ("B", lambda_elec_B, lambda_vdw_B),
        ):
            for idx, (e, v) in enumerate(zip(elec, vdw)):
                if e < 1 and v == 1:
                    raise ValueError(
                        "There are states along this lambda schedule where "
                        "there are atoms with charges but no LJ interactions: "
                        f"State {state}: lambda {idx}: elec {e} vdW {v}"
                    )

    def _validate(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None,
        extends: gufe.ProtocolDAGResult | None = None,
    ) -> None:
        # Check we're not trying to extend
        if extends:
            # This technically should be NotImplementedError
            # but gufe.Protocol.validate calls `_validate` wrapped
            # around a try/except for that error type
            raise ValueError("Can't extend simulations yet")

        # Check the mappping
        if mapping is not None:
            wmsg = "A mapping was passed but is not used by this Protocol"
            warnings.warn(wmsg)

        # Validate end states
        system_validation.validate_chemical_system(stateA)
        system_validation.validate_chemical_system(stateB)
        self._validate_endstates(stateA, stateB)

        # Validate the lambda schedule
        self._validate_lambda_schedule(
            self.settings.solvent_lambda_settings,
            self.settings.solvent_simulation_settings,
        )

        self._validate_lambda_schedule(
            self.settings.complex_lambda_settings,
            self.settings.complex_simulation_settings,
        )

        # Check nonbonded and solvent compatibility
        nonbonded_method = self.settings.forcefield_settings.nonbonded_method
        # Validate solvent component
        system_validation.validate_solvent(stateA, nonbonded_method)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(
            self.settings.solvent_solvation_settings
        )
        settings_validation.validate_openmm_solvation_settings(
            self.settings.complex_solvation_settings
        )

        # Validate the barostat used in combination with the protein component
        system_validation.validate_barostat(
            stateA, self.settings.complex_integrator_settings.barostat
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ) -> list[gufe.ProtocolUnit]:
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # Get the alchemical components
        alchem_comps = system_validation.get_alchemical_components(
            stateA,
            stateB,
        )

        # Create list units for complex and solvent transforms
        alchname_A = alchem_comps["stateA"][0].name
        alchname_B = alchem_comps["stateB"][0].name

        unit_classes: dict[str, dict[str, type[gufe.ProtocolUnit]]] = {
            "solvent": {
                "setup": SepTopSolventSetupUnit,
                "simulation": SepTopSolventRunUnit,
                "analysis": SepTopSolventAnalysisUnit,
            },
            "complex": {
                "setup": SepTopComplexSetupUnit,
                "simulation": SepTopComplexRunUnit,
                "analysis": SepTopComplexAnalysisUnit,
            },
        }

        protocol_units: dict[str, list[gufe.ProtocolUnit]] = {"solvent": [], "complex": []}

        for i in range(self.settings.protocol_repeats):
            repeat_id = int(uuid.uuid4())
            for phase in ["solvent", "complex"]:
                setup = unit_classes[phase]["setup"](
                    protocol=self,
                    stateA=stateA,
                    stateB=stateB,
                    alchemical_components=alchem_comps,
                    generation=0,
                    repeat_id=repeat_id,
                    name=(
                        f"SepTop RBFE Setup, transformation {alchname_A} to "
                        f"{alchname_B}, {phase} leg: repeat {i} generation 0"
                    ),
                )

                simulation = unit_classes[phase]["simulation"](
                    protocol=self,
                    stateA=stateA,
                    stateB=stateB,
                    alchemical_components=alchem_comps,
                    setup=setup,
                    generation=0,
                    repeat_id=repeat_id,
                    name=(
                        f"SepTop RBFE Run, transformation {alchname_A} to "
                        f"{alchname_B}, {phase} leg: repeat {i} generation 0"
                    ),
                )

                analysis = unit_classes[phase]["analysis"](
                    protocol=self,
                    setup=setup,
                    simulation=simulation,
                    generation=0,
                    repeat_id=repeat_id,
                    name=(
                        f"SepTop RBFE Analysis, transformation {alchname_A} to "
                        f"{alchname_B}, {phase} leg: repeat {i} generation 0"
                    ),
                )

                protocol_units[phase] += [setup, simulation, analysis]

        return protocol_units["solvent"] + protocol_units["complex"]

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, dict[str, Any]]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        unsorted_solvent_repeats = defaultdict(list)
        unsorted_complex_repeats = defaultdict(list)

        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if ("Analysis" not in pu.name) or (not pu.ok()):
                    continue
                if pu.outputs["simtype"] == "solvent":
                    unsorted_solvent_repeats[pu.outputs["repeat_id"]].append(pu)
                else:
                    unsorted_complex_repeats[pu.outputs["repeat_id"]].append(pu)

        repeats: dict[str, dict[str, list[gufe.ProtocolUnitResult]]] = {
            "solvent": {},
            "complex": {},
        }

        for k, v in unsorted_solvent_repeats.items():
            repeats["solvent"][str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        for k, v in unsorted_complex_repeats.items():
            repeats["complex"][str(k)] = sorted(v, key=lambda x: x.outputs["generation"])
        return repeats
