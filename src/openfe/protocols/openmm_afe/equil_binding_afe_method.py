# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium Binding AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_binding_afe_method`
==========================================================================================================

This module implements the necessary methodology tooling to calculate an
absolute binding free energy using OpenMM tools and one of the following
alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------
* Alchemical species with a net charge are not currently supported.
* Disapearing molecules are only allowed in state A.
* Only small molecules are allowed to act as alchemical molecules.

Acknowledgements
----------------
* This Protocol re-implements components from
  `Yank <https://github.com/choderalab/yank>`_.

"""

import logging
import uuid
import warnings
from collections import defaultdict
from typing import Any, Iterable

import gufe
from gufe import (
    BaseSolventComponent,
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    settings,
)
from openff.units import unit as offunit

from openfe.due import Doi, due
from openfe.protocols.openmm_afe.equil_afe_settings import (
    ABFEPreEquilOutputSettings,
    AbsoluteBindingSettings,
    AlchemicalSettings,
    BoreschRestraintSettings,
    IntegratorSettings,
    LambdaSettings,
    MDSimulationSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)
from openfe.protocols.openmm_utils import (
    settings_validation,
    system_validation,
)

from .abfe_units import (
    ABFEComplexAnalysisUnit,
    ABFEComplexSetupUnit,
    ABFEComplexSimUnit,
    ABFESolventAnalysisUnit,
    ABFESolventSetupUnit,
    ABFESolventSimUnit,
)
from .afe_protocol_results import AbsoluteBindingProtocolResult

due.cite(
    Doi("10.5281/zenodo.596504"),
    description="Yank",
    path="openfe.protocols.openmm_afe.equil_binding_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.5281/zenodo.596622"),
    description="OpenMMTools",
    path="openfe.protocols.openmm_afe.equil_binding_afe_method",
    cite_module=True,
)

due.cite(
    Doi("10.1371/journal.pcbi.1005659"),
    description="OpenMM",
    path="openfe.protocols.openmm_afe.equil_binding_afe_method",
    cite_module=True,
)


logger = logging.getLogger(__name__)


class AbsoluteBindingProtocol(gufe.Protocol):
    """
    Absolute binding free energy calculations using OpenMM and OpenMMTools.

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingSettings`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingProtocolResult`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingSolventUnit`
    :class:`openfe.protocols.openmm_afe.AbsoluteBindingComplexUnit`
    """

    result_cls = AbsoluteBindingProtocolResult
    _settings_cls = AbsoluteBindingSettings
    _settings: AbsoluteBindingSettings

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
        # fmt: off
        return AbsoluteBindingSettings(
            protocol_repeats=3,
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            alchemical_settings=AlchemicalSettings(),
            solvent_lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0, 0.25, 0.5, 0.75, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                ],
                lambda_vdw=[
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0
                ],
                lambda_restraints=[
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ],
            ),
            complex_lambda_settings=LambdaSettings(
                lambda_elec=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0
                ],
                lambda_vdw=[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
                ],
                lambda_restraints=[
                    0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0, 1.00, 1.0
                ],
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            complex_solvation_settings=OpenMMSolvationSettings(
                solvent_padding=1.0 * offunit.nanometer,
            ),
            solvent_solvation_settings=OpenMMSolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            solvent_integrator_settings=IntegratorSettings(),
            complex_integrator_settings=IntegratorSettings(),
            restraint_settings=BoreschRestraintSettings(),
            solvent_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.1 * offunit.nanosecond,
                equilibration_length=0.2 * offunit.nanosecond,
                production_length=0.5 * offunit.nanosecond,
            ),
            solvent_equil_output_settings=ABFEPreEquilOutputSettings(),
            solvent_simulation_settings=MultiStateSimulationSettings(
                n_replicas=14,
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            solvent_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename="solvent.nc",
                checkpoint_storage_filename="solvent_checkpoint.nc",
            ),
            complex_equil_simulation_settings=MDSimulationSettings(
                equilibration_length_nvt=0.25 * offunit.nanosecond,
                equilibration_length=0.5 * offunit.nanosecond,
                production_length=5.0 * offunit.nanosecond,
            ),
            complex_equil_output_settings=ABFEPreEquilOutputSettings(),
            complex_simulation_settings=MultiStateSimulationSettings(
                n_replicas=30,
                equilibration_length=1 * offunit.nanosecond,
                production_length=10.0 * offunit.nanosecond,
            ),
            complex_output_settings=MultiStateOutputSettings(
                output_structure="alchemical_system.pdb",
                output_filename="complex.nc",
                checkpoint_storage_filename="complex_checkpoint.nc",
            ),
        )
        # fmt: on

    @staticmethod
    def _validate_endstates(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
    ) -> None:
        """
        A binding transformation is defined (in terms of gufe components)
        as starting from one or more ligands with one protein and solvent,
        that then ends up in a state with one less ligand.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If stateA & stateB do not contain a ProteinComponent.
          If stateA & stateB do not contain a SolventComponent.
          If stateA has more than one unique Component.
          If the stateA unique Component is not a SmallMoleculeComponent.
          If stateB contains any unique Components.
          If the alchemical species is charged.
        """
        if not (stateA.contains(ProteinComponent) and stateB.contains(ProteinComponent)):
            errmsg = "No ProteinComponent found"
            raise ValueError(errmsg)

        if not (stateA.contains(SolventComponent) and stateB.contains(SolventComponent)):
            errmsg = "No SolventComponent found"
            raise ValueError(errmsg)

        # Needs gufe 1.3
        diff = stateA.component_diff(stateB)
        if len(diff[0]) != 1:
            errmsg = (
                "Only one alchemical species is supported. "
                f"Number of unique components found in stateA: {len(diff[0])}."
            )
            raise ValueError(errmsg)

        if not isinstance(diff[0][0], SmallMoleculeComponent):
            errmsg = (
                "Only dissapearing small molecule components "
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
          the settings for either the complex or solvent phase

        Raises
        ------
        ValueError
          If the number of lambda windows differs for electrostatics, sterics,
          and restraints.
          If the number of replicas does not match the number of lambda windows.
          If there are states with naked charges.
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

        # Check if there are no lambda windows with naked charges
        for inx, lam in enumerate(lambda_elec):
            if lam < 1 and lambda_vdw[inx] == 1:
                errmsg = (
                    "There are states along this lambda schedule "
                    "where there are atoms with charges but no LJ "
                    f"interactions: lambda {inx}: "
                    f"elec {lam} vdW {lambda_vdw[inx]}"
                )
                raise ValueError(errmsg)

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ):
        # Check we're not extending
        if extends is not None:
            # This technically should be NotImplementedError
            # but gufe.Protocol.validate calls `_validate` wrapped around an
            # except for NotImplementedError, so we can't raise it here
            raise ValueError("Can't extend simulations yet")

        # Check we're not using a mapping, since we're not doing anything with it
        if mapping is not None:
            wmsg = "A mapping was passed but is not used by this Protocol."
            warnings.warn(wmsg)

        # Validate the end states & alchemical components
        self._validate_endstates(stateA, stateB)

        # Validate the complex lambda schedule
        self._validate_lambda_schedule(
            self.settings.complex_lambda_settings,
            self.settings.complex_simulation_settings,
        )

        # If the complex restraints schedule is all zero, it might be bad
        # but we don't dissallow it.
        if all([i == 0.0 for i in self.settings.complex_lambda_settings.lambda_restraints]):
            wmsg = (
                "No restraints are being applied in the complex phase, "
                "this will likely lead to problematic results."
            )
            warnings.warn(wmsg)

        # Validate the solvent lambda schedule
        self._validate_lambda_schedule(
            self.settings.solvent_lambda_settings,
            self.settings.solvent_simulation_settings,
        )

        # If the solvent restraints schedule is not all one, it was likely
        # copied from the complex schedule. In this case we just ignore
        # the values and let the user know.
        # P.S. we don't need to change the settings at this point
        # the list gets popped out later in the SolventUnit, because we
        # don't have a restraint parameter state.

        if any([i != 0.0 for i in self.settings.solvent_lambda_settings.lambda_restraints]):
            wmsg = (
                "There is an attempt to add restraints in the solvent "
                "phase. This protocol does not apply restraints in the "
                "solvent phase. These restraint lambda values will be ignored."
            )
            warnings.warn(wmsg)

        # Check nonbond & solvent compatibility
        nonbonded_method = self.settings.forcefield_settings.nonbonded_method
        # Use the more complete system validation solvent checks
        system_validation.validate_solvent(stateA, nonbonded_method)

        # Validate the barostat used in combination with the protein component
        system_validation.validate_protein_barostat(
            stateA, self.settings.complex_integrator_settings.barostat
        )

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(
            self.settings.solvent_solvation_settings
        )
        settings_validation.validate_openmm_solvation_settings(
            self.settings.complex_solvation_settings
        )

        # Validate integrator things
        # We validate the timstep for both the complex & solvent settings
        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.complex_integrator_settings.timestep,
        )

        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.solvent_integrator_settings.timestep,
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
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
            "solvent": {
                "setup": ABFESolventSetupUnit,
                "simulation": ABFESolventSimUnit,
                "analysis": ABFESolventAnalysisUnit,
            },
            "complex": {
                "setup": ABFEComplexSetupUnit,
                "simulation": ABFEComplexSimUnit,
                "analysis": ABFEComplexAnalysisUnit,
            },
        }

        protocol_units: dict[str, list[gufe.ProtocolUnit]] = {"solvent": [], "complex": []}

        for phase in ["solvent", "complex"]:
            for i in range(self.settings.protocol_repeats):
                repeat_id = int(uuid.uuid4())

                setup = unit_classes[phase]["setup"](
                    protocol=self,
                    stateA=stateA,
                    stateB=stateB,
                    alchemical_components=alchem_comps,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"ABFE Setup: {alchname} {phase} leg: repeat {i} generation 0",
                )

                simulation = unit_classes[phase]["simulation"](
                    protocol=self,
                    # only need state A & alchem comps
                    stateA=stateA,
                    alchemical_components=alchem_comps,
                    setup_results=setup,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"ABFE Simulation: {alchname} {phase} leg: repeat {i} generation 0",
                )

                analysis = unit_classes[phase]["analysis"](
                    protocol=self,
                    setup_results=setup,
                    simulation_results=simulation,
                    generation=0,
                    repeat_id=repeat_id,
                    name=f"ABFE Analysis: {alchname} {phase} leg, repeat {i} generation 0",
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
