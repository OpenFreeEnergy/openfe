# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium Solvation AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_solvation_afe_method`
===============================================================================================================

This module implements the necessary methodology tooling to run calculate an
absolute solvation free energy using OpenMM tools and one of the following
alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------
* Alchemical species with a net charge are not currently supported.
* Disapearing molecules are only allowed in state A. Support for
  appearing molecules will be added in due course.
* Only small molecules are allowed to act as alchemical molecules.
  Alchemically changing protein or solvent components would induce
  perturbations which are too large to be handled by this Protocol.


Acknowledgements
----------------
* Originally based on hydration.py in
  `espaloma_charge <https://github.com/choderalab/espaloma_charge>`_

"""
import logging
import uuid
import warnings
from collections import defaultdict
from typing import Any, Iterable, Optional, Union

import gufe
import numpy as np
from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    settings,
)
from openff.units import unit as offunit

from openfe.due import Doi, due
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteSolvationSettings,
    AlchemicalSettings,
    IntegratorSettings,
    LambdaSettings,
    MDOutputSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)

from ..openmm_utils import settings_validation, system_validation
from .ahfe_units import (
    AHFESolventSetupUnit,
    AHFESolventSimUnit,
    AHFESolventAnalysisUnit,
    AHFEVacuumSetupUnit,
    AHFEVacuumSimUnit,
    AHFEVacuumAnalysisUnit,
)
from .afe_protocol_results import AbsoluteSolvationProtocolResult


due.cite(
    Doi("10.5281/zenodo.596504"),
    description="Yank",
    path="openfe.protocols.openmm_afe.equil_solvation_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.48550/ARXIV.2302.06758"),
    description="EspalomaCharge",
    path="openfe.protocols.openmm_afe.equil_solvation_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.5281/zenodo.596622"),
    description="OpenMMTools",
    path="openfe.protocols.openmm_afe.equil_solvation_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.1371/journal.pcbi.1005659"),
    description="OpenMM",
    path="openfe.protocols.openmm_afe.equil_solvation_afe_method",
    cite_module=True,
)


logger = logging.getLogger(__name__)


class AbsoluteSolvationProtocol(gufe.Protocol):
    """
    Absolute solvation free energy calculations using OpenMM and OpenMMTools.

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_afe.AbsoluteSolvationSettings`
    :class:`openfe.protocols.openmm_afe.AbsoluteSolvationProtocolResult`
    :class:`openfe.protocols.openmm_afe.AbsoluteSolvationVacuumUnit`
    :class:`openfe.protocols.openmm_afe.AbsoluteSolvationSolventUnit`
    """

    result_cls = AbsoluteSolvationProtocolResult
    _settings_cls = AbsoluteSolvationSettings
    _settings: AbsoluteSolvationSettings

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
        return AbsoluteSolvationSettings(
            protocol_repeats=3,
            solvent_forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            vacuum_forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(
                nonbonded_method="nocutoff",
            ),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
            lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                lambda_vdw=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.12, 0.24,
                    0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0],
                lambda_restraints=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=OpenMMSolvationSettings(),
            vacuum_engine_settings=OpenMMEngineSettings(),
            solvent_engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            solvent_equil_output_settings=MDOutputSettings(
                equil_nvt_structure="equil_nvt_structure.pdb",
                equil_npt_structure="equil_npt_structure.pdb",
                production_trajectory_filename="production_equil.xtc",
                log_output="equil_simulation.log",
            ),
            solvent_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_filename="solvent.nc",
                checkpoint_storage_filename="solvent_checkpoint.nc",
            ),
            vacuum_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=None,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            vacuum_equil_output_settings=MDOutputSettings(
                equil_nvt_structure=None,
                equil_npt_structure="equil_structure.pdb",
                production_trajectory_filename="production_equil.xtc",
                log_output="equil_simulation.log",
            ),
            vacuum_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=0.5 * offunit.nanosecond,
                production_length=2.0 * offunit.nanosecond,
            ),
            vacuum_output_settings=MultiStateOutputSettings(
                output_filename="vacuum.nc",
                checkpoint_storage_filename="vacuum_checkpoint.nc",
            ),
        )  # fmt: skip

    @staticmethod
    def _validate_endstates(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
    ) -> None:
        """
        A solvent transformation is defined (in terms of gufe components)
        as starting from one or more ligands in solvent and
        ending up in a state with one less ligand.

        No protein components are allowed.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If stateA or stateB contains a ProteinComponent.
          If there is no SolventComponent in either stateA or stateB.
          If there are alchemical components in state B.
          If there are non SmallMoleculeComponent alchemical species.
          If there are more than one alchemical species.
          If the alchemical species is charged.

        Notes
        -----
        * Currently doesn't support alchemical components in state B.
        * Currently doesn't support alchemical components which are not
          SmallMoleculeComponents.
        * Currently doesn't support more than one alchemical component
          being desolvated.
        * Currently doesn't support charged alchemical components.
        * Solvent must always be present in both end states.
        """
        # Check that there are no protein components
        if stateA.contains(ProteinComponent) or stateB.contains(ProteinComponent):
            errmsg = "Protein components are not allowed for absolute solvation free energies."
            raise ValueError(errmsg)

        # Check that there is a solvent component in both end states
        if not (stateA.contains(SolventComponent) and stateB.contains(SolventComponent)):
            errmsg = "No SolventComponent found in stateA and/or stateB"
            raise ValueError(errmsg)

        # Now we check the alchemical Components
        diff = stateA.component_diff(stateB)

        # Check that there's only one state A unique Component
        if len(diff[0]) != 1:
            errmsg = (
                "Only one alchemical species is supported "
                "for absolute solvation free energies. "
                f"Number of unique components found in stateA: {len(diff[0])}."
            )
            raise ValueError(errmsg)

        # Make sure that the state A unique is an SMC
        if not isinstance(diff[0][0], SmallMoleculeComponent):
            errmsg = (
                "Only dissapearing SmallMoleculeComponents "
                "are supported by this protocol. "
                f"Found a {type(diff[0][0])}"
            )
            raise ValueError(errmsg)

        # Check that the state A unique isn't charged
        if diff[0][0].total_charge != 0:
            errmsg = (
                "Charged alchemical molecules are not currently "
                "supported for solvation free energies. "
                f"Molecule total charge: {diff[0][0].total_charge}."
            )
            raise ValueError(errmsg)

        # If there are any alchemical Components in state B
        if len(diff[1]) > 0:
            errmsg = "Components appearing in state B are not currently supported"
            raise ValueError(errmsg)

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
          the settings for either the vacuum or solvent phase

        Raises
        ------
        ValueError
          If the number of lambda windows differs for electrostatics and sterics.
          If the number of replicas does not match the number of lambda windows.
          If there are states with naked charges.
        Warnings
          If there are non-zero values for restraints (lambda_restraints).
        """

        lambda_elec = lambda_settings.lambda_elec
        lambda_vdw = lambda_settings.lambda_vdw
        lambda_restraints = lambda_settings.lambda_restraints
        n_replicas = simulation_settings.n_replicas

        # Ensure that all lambda components have equal amount of windows
        lambda_components = [lambda_vdw, lambda_elec, lambda_restraints]
        it = iter(lambda_components)
        the_len = len(next(it))
        if not all(len(lambda_comp) == the_len for lambda_comp in it):
            errmsg = (
                "Components elec, vdw, and restraints must have equal amount"
                f" of lambda windows. Got {len(lambda_elec)} elec lambda"
                f" windows, {len(lambda_vdw)} vdw lambda windows, and"
                f"{len(lambda_restraints)} restraints lambda windows."
            )
            raise ValueError(errmsg)

        # Ensure that number of overall lambda windows matches number of lambda
        # windows for individual components
        if n_replicas != len(lambda_vdw):
            errmsg = (
                f"Number of replicas {n_replicas} does not equal the"
                f" number of lambda windows {len(lambda_vdw)}"
            )
            raise ValueError(errmsg)

        # Check if there are lambda windows with naked charges
        for inx, lam in enumerate(lambda_elec):
            if lam < 1 and lambda_vdw[inx] == 1:
                errmsg = (
                    "There are states along this lambda schedule "
                    "where there are atoms with charges but no LJ "
                    f"interactions: lambda {inx}: "
                    f"elec {lam} vdW {lambda_vdw[inx]}"
                )
                raise ValueError(errmsg)

        # Check if there are lambda windows with non-zero restraints
        if len([r for r in lambda_restraints if r != 0]) > 0:
            wmsg = (
                "Non-zero restraint lambdas applied. The absolute "
                "solvation protocol doesn't apply restraints, "
                "therefore restraints won't be applied. "
                f"Given lambda_restraints: {lambda_restraints}"
            )
            logger.warning(wmsg)
            warnings.warn(wmsg)

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ):
        # Check we're not extending
        if extends is not None:
            # This should be a NotImplementedError, but the underlying
            # `validate` method wraps a call to `_validate` around a
            # NotImplementedError exception guard
            raise ValueError("Can't extend simulations yet")

        # Check we're not using a mapping, since we're not doing anything with it
        if mapping is not None:
            wmsg = "A mapping was passed but is not used by this Protocol."
            warnings.warn(wmsg)

        # Validate the endstates & alchemical components
        self._validate_endstates(stateA, stateB)

        # Validate the lambda schedule
        for solv_sets in (
            self.settings.solvent_simulation_settings,
            self.settings.vacuum_simulation_settings,
        ):
            self._validate_lambda_schedule(
                self.settings.lambda_settings,
                solv_sets,
            )

        # Check nonbond & solvent compatibility
        solv_nonbonded_method = self.settings.solvent_forcefield_settings.nonbonded_method
        vac_nonbonded_method = self.settings.vacuum_forcefield_settings.nonbonded_method

        # Use the more complete system validation solvent checks
        system_validation.validate_solvent(stateA, solv_nonbonded_method)

        # Gas phase is always gas phase
        if vac_nonbonded_method.lower() != "nocutoff":
            errmsg = (
                "Only the nocutoff nonbonded_method is supported for "
                f"vacuum calculations, {vac_nonbonded_method} was "
                "passed"
            )
            raise ValueError(errmsg)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(self.settings.solvation_settings)

        # Check vacuum equilibration MD settings is 0 ns
        nvt_time = self.settings.vacuum_equil_simulation_settings.equilibration_length_nvt
        if nvt_time is not None:
            if not np.allclose(nvt_time, 0 * offunit.nanosecond):
                errmsg = "NVT equilibration cannot be run in vacuum simulation"
                raise ValueError(errmsg)

        # Validate integrator things
        settings_validation.validate_timestep(
            self.settings.vacuum_forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

        settings_validation.validate_timestep(
            self.settings.solvent_forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # Validate inputs
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # Get the alchemical components
        alchem_comps = system_validation.get_alchemical_components(
            stateA,
            stateB,
        )

        # Get the name of the alchemical species
        alchname = alchem_comps["stateA"][0].name

        unit_classes = {
            'solvent': {
                'setup': AHFESolventSetupUnit,
                'simulation': AHFESolventSimUnit,
                'analysis': AHFESolventAnalysisUnit,
            },
            'vacuum': {
                'setup': AHFEVacuumSetupUnit,
                'simulation': AHFEVacuumSimUnit,
                'analysis': AHFEVacuumAnalysisUnit,
            }
        }

        protocol_units = {'solvent': [], 'vacuum': []}

        for phase in ['solvent', 'vacuum']:
            for i in range(self.settings.protocol_repeats):
                repeat_id = int(uuid.uuid4())

                setup = unit_classes[phase]['setup'](
                    protocol=self,
                    stateA=stateA,
                    stateB=stateB,
                    alchemical_components=alchem_comps,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"Absolute Hydration Setup: {alchname} solvent leg: repeat {i} generation 0",
                )

                simulation = unit_classes[phase]['simulation'](
                    protocol=self,
                    # only need state A & alchem comps
                    stateA=stateA,
                    alchemical_components=alchem_comps,
                    setup_results=setup,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"Absolute Hydration Simulation: {alchname} solvent leg: repeat {i} generation 0",
                )

                analysis = unit_classes[phase]['analysis'](
                    protocol=self,
                    setup_results=setup,
                    simulation_results=simulation,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"Absolute Hydration Analysis: {alchname} solvent leg, repeat {i} generation 0",
                )

                protocol_units[phase] += [setup, simulation, analysis]

        return protocol_units["solvent"] + protocol_units["vacuum"]

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, dict[str, Any]]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        unsorted_solvent_repeats = defaultdict(list)
        unsorted_vacuum_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if ("Analysis" not in pu.name) or (not pu.ok()):
                    continue
                if pu.outputs["simtype"] == "solvent":
                    unsorted_solvent_repeats[pu.outputs["repeat_id"]].append(pu)
                else:
                    unsorted_vacuum_repeats[pu.outputs["repeat_id"]].append(pu)

        repeats: dict[str, dict[str, list[gufe.ProtocolUnitResult]]] = {
            "solvent": {},
            "vacuum": {},
        }
        for k, v in unsorted_solvent_repeats.items():
            repeats["solvent"][str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        for k, v in unsorted_vacuum_repeats.items():
            repeats["vacuum"][str(k)] = sorted(v, key=lambda x: x.outputs["generation"])
        return repeats
