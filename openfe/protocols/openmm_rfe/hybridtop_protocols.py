# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Hybrid Topology Protocols using OpenMM and OpenMMTools in a Perses-like manner.

Acknowledgements
----------------
These Protocols are based on, and leverages components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections import defaultdict
from typing import Any, Iterable, Optional, Union
import numpy as np

import gufe
from gufe import (
    ChemicalSystem,
    Component,
    ComponentMapping,
    LigandAtomMapping,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    settings,
)
from openff.units import unit as offunit

from openfe.due import Doi, due

from ..openmm_utils import (
    settings_validation,
    system_validation,
)
from .equil_rfe_settings import (
    AlchemicalSettings,
    IntegratorSettings,
    LambdaSettings,
    MultiStateOutputSettings,
    MultiStateSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    RelativeHybridTopologyProtocolSettings,
)
from .hybridtop_protocol_results import RelativeHybridTopologyProtocolResult
from .hybridtop_units import RelativeHybridTopologyProtocolUnit


logger = logging.getLogger(__name__)


due.cite(
    Doi("10.5281/zenodo.1297683"),
    description="Perses",
    path="openfe.protocols.openmm_rfe.hybridtop_protocols",
    cite_module=True,
)

due.cite(
    Doi("10.5281/zenodo.596622"),
    description="OpenMMTools",
    path="openfe.protocols.openmm_rfe.hybridtop_protocols",
    cite_module=True,
)

due.cite(
    Doi("10.1371/journal.pcbi.1005659"),
    description="OpenMM",
    path="openfe.protocols.openmm_rfe.hybridtop_protocols",
    cite_module=True,
)


class RelativeHybridTopologyProtocol(gufe.Protocol):
    """
    Relative Free Energy calculations using OpenMM and OpenMMTools.

    Based on `Perses <https://github.com/choderalab/perses>`_

    See Also
    --------
    :mod:`openfe.protocols`
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologySettings`
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyResult`
    :class:`openfe.protocols.openmm_rfe.RelativeHybridTopologyProtocolUnit`
    """

    result_cls = RelativeHybridTopologyProtocolResult
    _settings_cls = RelativeHybridTopologyProtocolSettings
    _settings: RelativeHybridTopologyProtocolSettings

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
        return RelativeHybridTopologyProtocolSettings(
            protocol_repeats=3,
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * offunit.kelvin,
                pressure=1 * offunit.bar,
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=OpenMMSolvationSettings(),
            alchemical_settings=AlchemicalSettings(softcore_LJ="gapsys"),
            lambda_settings=LambdaSettings(),
            simulation_settings=MultiStateSimulationSettings(
                equilibration_length=1.0 * offunit.nanosecond,
                production_length=5.0 * offunit.nanosecond,
            ),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            output_settings=MultiStateOutputSettings(),
        )

    @classmethod
    def _adaptive_settings(
        cls,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.LigandAtomMapping | list[gufe.LigandAtomMapping],
        initial_settings: None | RelativeHybridTopologyProtocolSettings = None,
    ) -> RelativeHybridTopologyProtocolSettings:
        """
        Get the recommended OpenFE settings for this protocol based on the input states involved in the
        transformation.

        These are intended as a suitable starting point for creating an instance of this protocol, which can be further
        customized before performing a Protocol.

        Parameters
        ----------
        stateA : ChemicalSystem
            The initial state of the transformation.
        stateB : ChemicalSystem
            The final state of the transformation.
        mapping : LigandAtomMapping | list[LigandAtomMapping]
            The mapping(s) between transforming components in stateA and stateB.
        initial_settings : None | RelativeHybridTopologyProtocolSettings, optional
            Initial settings to base the adaptive settings on. If None, default settings are used.

        Returns
        -------
        RelativeHybridTopologyProtocolSettings
            The recommended settings for this protocol based on the input states.

        Notes
        -----
        - If the transformation involves a change in net charge, the settings are adapted to use a more expensive
          protocol with 22 lambda windows and 20 ns production length per window.
        - If both states contain a ProteinComponent, the solvation padding is set to 1 nm.
        - If initial_settings is provided, the adaptive settings are based on a copy of these settings.
        """
        # use initial settings or default settings
        # this is needed for the CLI so we don't override user settings
        if initial_settings is not None:
            protocol_settings = initial_settings.copy(deep=True)
        else:
            protocol_settings = cls.default_settings()

        if isinstance(mapping, list):
            mapping = mapping[0]

        if mapping.get_alchemical_charge_difference() != 0:
            # apply the recommended charge change settings taken from the industry benchmarking as fast settings not validated
            # <https://github.com/OpenFreeEnergy/IndustryBenchmarks2024/blob/2df362306e2727321d55d16e06919559338c4250/industry_benchmarks/utils/plan_rbfe_network.py#L128-L146>
            info = (
                "Charge changing transformation between ligands "
                f"{mapping.componentA.name} and {mapping.componentB.name}. "
                "A more expensive protocol with 22 lambda windows, sampled "
                "for 20 ns each, will be used here."
            )
            logger.info(info)
            protocol_settings.alchemical_settings.explicit_charge_correction = True
            protocol_settings.simulation_settings.production_length = 20 * offunit.nanosecond
            protocol_settings.simulation_settings.n_replicas = 22
            protocol_settings.lambda_settings.lambda_windows = 22

        # adapt the solvation padding based on the system components
        if stateA.contains(ProteinComponent) and stateB.contains(ProteinComponent):
            protocol_settings.solvation_settings.solvent_padding = 1 * offunit.nanometer

        return protocol_settings

    @staticmethod
    def _validate_endstates(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
    ) -> None:
        """
        Validates the end states for the RFE protocol.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A.
        stateB : ChemicalSystem
          The chemical system of end state B.

        Raises
        ------
        ValueError
          * If either state contains more than one unique Component.
          * If unique components are not SmallMoleculeComponents.
        """
        # Get the difference in Components between each state
        diff = stateA.component_diff(stateB)

        for i, entry in enumerate(diff):
            state_label = "A" if i == 0 else "B"

            # Check that there is only one unique Component in each state
            if len(entry) != 1:
                errmsg = (
                    "Only one alchemical component is allowed per end state. "
                    f"Found {len(entry)} in state {state_label}."
                )
                raise ValueError(errmsg)

            # Check that the unique Component is a SmallMoleculeComponent
            if not isinstance(entry[0], SmallMoleculeComponent):
                errmsg = (
                    f"Alchemical component in state {state_label} is of type "
                    f"{type(entry[0])}, but only SmallMoleculeComponents "
                    "transformations are currently supported."
                )
                raise ValueError(errmsg)

    @staticmethod
    def _validate_mapping(
        mapping: Optional[Union[ComponentMapping, list[ComponentMapping]]],
        alchemical_components: dict[str, list[Component]],
    ) -> None:
        """
        Validates that the provided mapping(s) are suitable for the RFE protocol.

        Parameters
        ----------
        mapping : Optional[Union[ComponentMapping, list[ComponentMapping]]]
          all mappings between transforming components.
        alchemical_components : dict[str, list[Component]]
          Dictionary contatining the alchemical components for
          states A and B.

        Raises
        ------
        ValueError
          * If there are more than one mapping or mapping is None
          * If the mapping components are not in the alchemical components.
        UserWarning
          * Mappings which involve element changes in core atoms
        """
        # if a single mapping is provided, convert to list
        if isinstance(mapping, ComponentMapping):
            mapping = [mapping]

        # For now we only support a single mapping
        if mapping is None or len(mapping) > 1:
            errmsg = "A single LigandAtomMapping is expected for this Protocol"
            raise ValueError(errmsg)

        # check that the mapping components are in the alchemical components
        for m in mapping:
            if m.componentA not in alchemical_components["stateA"]:
                raise ValueError(f"Mapping componentA {m.componentA} not in alchemical components of stateA")
            if m.componentB not in alchemical_components["stateB"]:
                raise ValueError(f"Mapping componentB {m.componentB} not in alchemical components of stateB")

        # TODO: remove - this is now the default behaviour?
        # Check for element changes in mappings
        for m in mapping:
            molA = m.componentA.to_rdkit()
            molB = m.componentB.to_rdkit()
            for i, j in m.componentA_to_componentB.items():
                atomA = molA.GetAtomWithIdx(i)
                atomB = molB.GetAtomWithIdx(j)
                if atomA.GetAtomicNum() != atomB.GetAtomicNum():
                    wmsg = (
                        f"Element change in mapping between atoms "
                        f"Ligand A: {i} (element {atomA.GetAtomicNum()}) and "
                        f"Ligand B: {j} (element {atomB.GetAtomicNum()})\n"
                        "No mass scaling is attempted in the hybrid topology, "
                        "the average mass of the two atoms will be used in the "
                        "simulation"
                    )
                    logger.warning(wmsg)
                    warnings.warn(wmsg)

    @staticmethod
    def _validate_smcs(
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
    ) -> None:
        """
        Validates the SmallMoleculeComponents.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A.
        stateB : ChemicalSystem
          The chemical system of end state B.

        Raises
        ------
        ValueError
          * If there are isomorphic SmallMoleculeComponents with
            different charges within a given ChemicalSystem.
        """
        smcs_A = stateA.get_components_of_type(SmallMoleculeComponent)
        smcs_B = stateB.get_components_of_type(SmallMoleculeComponent)
        smcs_all = list(set(smcs_A).union(set(smcs_B)))
        offmols = [m.to_openff() for m in smcs_all]

        def _equal_charges(moli, molj):
            # Base case, both molecules don't have charges
            if (moli.partial_charges is None) & (molj.partial_charges is None):
                return True
            # If either is None but not the other
            if (moli.partial_charges is None) ^ (molj.partial_charges is None):
                return False
            # Check if the charges are close to each other
            return np.allclose(moli.partial_charges, molj.partial_charges)

        clashes = []

        for smcs in [smcs_A, smcs_B]:
            offmols = [m.to_openff() for m in smcs]
            for i, moli in enumerate(offmols):
                for molj in offmols:
                    if moli.is_isomorphic_with(molj):
                        if not _equal_charges(moli, molj):
                            clashes.append(smcs[i])

        if len(clashes) > 0:
            errmsg = (
                "Found SmallMoleculeComponents are are isomorphic "
                "but with different charges, this is not currently allowed. "
                f"Affected components: {clashes}"
            )
            raise ValueError(errmsg)

    @staticmethod
    def _validate_charge_difference(
        mapping: LigandAtomMapping,
        nonbonded_method: str,
        explicit_charge_correction: bool,
        solvent_component: SolventComponent | None,
    ):
        """
        Validates the net charge difference between the two states.

        Parameters
        ----------
        mapping : dict[str, ComponentMapping]
          Dictionary of mappings between transforming components.
        nonbonded_method : str
          The OpenMM nonbonded method used for the simulation.
        explicit_charge_correction : bool
          Whether or not to use an explicit charge correction.
        solvent_component : openfe.SolventComponent | None
          The SolventComponent of the simulation.

        Raises
        ------
        ValueError
          * If an explicit charge correction is attempted and the
            nonbonded method is not PME.
          * If the absolute charge difference is greater than one
            and an explicit charge correction is attempted.
        UserWarning
          * If there is any charge difference.
        """
        difference = mapping.get_alchemical_charge_difference()

        if abs(difference) == 0:
            return

        if not explicit_charge_correction:
            wmsg = (
                f"A charge difference of {difference} is observed "
                "between the end states. No charge correction has "
                "been requested, please account for this in your "
                "final results."
            )
            logger.warning(wmsg)
            warnings.warn(wmsg)
            return

        if solvent_component is None:
            errmsg = "Cannot use eplicit charge correction without solvent"
            raise ValueError(errmsg)

        # We implicitly check earlier that we have to have pme for a solvated
        # system, so we only need to check the nonbonded method here
        if nonbonded_method.lower() != "pme":
            errmsg = "Explicit charge correction when not using PME is not currently supported."
            raise ValueError(errmsg)

        if abs(difference) > 1:
            errmsg = (
                f"A charge difference of {difference} is observed "
                "between the end states and an explicit charge  "
                "correction has been requested. Unfortunately "
                "only absolute differences of 1 are supported."
            )
            raise ValueError(errmsg)

        ion = {
            -1: solvent_component.positive_ion,
            1: solvent_component.negative_ion
        }[difference]

        wmsg = (
            f"A charge difference of {difference} is observed "
            "between the end states. This will be addressed by "
            f"transforming a water into a {ion} ion"
        )
        logger.info(wmsg)

    @staticmethod
    def _validate_simulation_settings(
        simulation_settings: MultiStateSimulationSettings,
        integrator_settings: IntegratorSettings,
        output_settings: MultiStateOutputSettings,
    ):
        """
        Validate various simulation settings, including but not limited to
        timestep conversions, and output file write frequencies.

        Parameters
        ----------
        simulation_settings : MultiStateSimulationSettings
          The sampler simulation settings.
        integrator_settings : IntegratorSettings
          Settings defining the behaviour of the integrator.
        output_settings : MultiStateOutputSettings
          Settings defining the simulation file writing behaviour.

        Raises
        ------
        ValueError
          * If the 
        """

        steps_per_iteration = settings_validation.convert_steps_per_iteration(
            simulation_settings=simulation_settings,
            integrator_settings=integrator_settings,
        )

        _ = settings_validation.get_simsteps(
            sim_length=simulation_settings.equilibration_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )

        _ = settings_validation.get_simsteps(
            sim_length=simulation_settings.production_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )

        _ = settings_validation.convert_checkpoint_interval_to_iterations(
            checkpoint_interval=output_settings.checkpoint_interval,
            time_per_iteration=simulation_settings.time_per_iteration,
        )

        if output_settings.positions_write_frequency is not None:
            _ = settings_validation.divmod_time_and_check(
                numerator=output_settings.positions_write_frequency,
                denominator=simulation_settings.time_per_iteration,
                numerator_name="output settings' positions_write_frequency",
                denominator_name="sampler settings' time_per_iteration",
            )

        if output_settings.velocities_write_frequency is not None:
            _ = settings_validation.divmod_time_and_check(
                numerator=output_settings.velocities_write_frequency,
                denominator=simulation_settings.time_per_iteration,
                numerator_name="output settings' velocities_write_frequency",
                denominator_name="sampler settings' time_per_iteration",
            )

        _, _ = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=simulation_settings,
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
            # but gufe.Protocol.validate calls `_validate` wrapped around an
            # except for NotImplementedError, so we can't raise it here
            raise ValueError("Can't extend simulations yet")

        # Validate the end states
        self._validate_endstates(stateA, stateB)

        # Valildate the mapping
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)
        self._validate_mapping(mapping, alchem_comps)

        # Validate the small molecule components
        self._validate_smcs(stateA, stateB)

        # Validate solvent component
        nonbond = self.settings.forcefield_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate solvation settings
        settings_validation.validate_openmm_solvation_settings(self.settings.solvation_settings)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # Validate charge difference
        # Note: validation depends on the mapping & solvent component checks
        if stateA.contains(SolventComponent):
            solv_comp = stateA.get_components_of_type(SolventComponent)[0]
        else:
            solv_comp = None

        self._validate_charge_difference(
            mapping=mapping[0] if isinstance(mapping, list) else mapping,
            nonbonded_method=self.settings.forcefield_settings.nonbonded_method,
            explicit_charge_correction=self.settings.alchemical_settings.explicit_charge_correction,
            solvent_component=solv_comp,
        )

        # Validate integrator things
        settings_validation.validate_timestep(
            self.settings.forcefield_settings.hydrogen_mass,
            self.settings.integrator_settings.timestep,
        )

        # Validate simulation & output settings
        self._validate_simulation_settings(
            self.settings.simulation_settings,
            self.settings.integrator_settings,
            self.settings.output_settings,
        )

        # Validate alchemical settings
        # PR #125 temporarily pin lambda schedule spacing to n_replicas
        if self.settings.simulation_settings.n_replicas != self.settings.lambda_settings.lambda_windows:
            errmsg = (
                "Number of replicas in ``simulation_settings``: "
                f"{self.settings.simulation_settings.n_replicas} must equal "
                "the number of lambda windows in lambda_settings: "
                f"{self.settings.lambda_settings.lambda_windows}."
            )
            raise ValueError(errmsg)

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]],
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # validate inputs
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        # get alchemical components and mapping
        alchem_comps = system_validation.get_alchemical_components(stateA, stateB)
        ligandmapping = mapping[0] if isinstance(mapping, list) else mapping

        # actually create and return Units
        Anames = ",".join(c.name for c in alchem_comps["stateA"])
        Bnames = ",".join(c.name for c in alchem_comps["stateB"])

        # our DAG has no dependencies, so just list units
        n_repeats = self.settings.protocol_repeats

        units = [
            RelativeHybridTopologyProtocolUnit(
                protocol=self,
                stateA=stateA,
                stateB=stateB,
                ligandmapping=ligandmapping,
                generation=0,
                repeat_id=int(uuid.uuid4()),
                name=f"{Anames} to {Bnames} repeat {i} generation 0",
            )
            for i in range(n_repeats)
        ]

        return units

    def _gather(self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]) -> dict[str, Any]:
        # result units will have a repeat_id and generations within this repeat_id
        # first group according to repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue

                unsorted_repeats[pu.outputs["repeat_id"]].append(pu)

        # then sort by generation within each repeat_id list
        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        # returns a dict of repeat_id: sorted list of ProtocolUnitResult
        return repeats
