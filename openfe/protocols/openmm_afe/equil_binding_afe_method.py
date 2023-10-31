# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium Binding AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_binding_afe_method`
==========================================================================================================

This module implements the necessary methodology tooling to run calculate an
absolute binding free energy using OpenMM tools and one of the following
alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling

Current limitations
-------------------
* Disapearing molecules are only allowed in state A. Support for
  appearing molecules will be added in due course.
* Only SmallMoleculeComponents are allowed to act as alchemical molecules.
  Support for ProteinComponents changing (e.g. peptides), may be supported
  in the future.
"""
from __future__ import annotations

import logging

from collections import defaultdict
import gufe
from gufe.components import Component
import itertools
import numpy as np
import numpy.typing as npt
from openff.units import unit
from typing import Dict, Optional, Union
from typing import Any, Iterable

from gufe import (
    settings,
    ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteBindingSettings, SystemSettings,
    SolvationSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings,
    SettingsBaseModel,
)
from ..openmm_utils import system_validation, settings_validation
from .base import BaseAbsoluteUnit
from openfe.utils import without_oechem_backend, log_system_probe


logger = logging.getLogger(__name__)


class AbsoluteBindingProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a AbsoluteBindingProtocol
    """
    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list
               in itertools.chain(self.data['complex'].values(), self.data['solvent'].values())):
            raise NotImplementedError("Can't stitch together results yet")

    def get_individual_estimates(self) -> dict[str, list[tuple[unit.Quantity, unit.Quantity]]]:
        """
        Get the individual estimate of the free energies.

        Returns
        -------
        dGs : dict[str, list[tuple[unit.Quantity, unit.Quantity]]]
          A dictionary, keyed `complex` and `solvent` for each leg
          of the thermodynamic cycle, with lists of tuples containing
          the individual free energy estimates and associated MBAR
          uncertainties for each repeat of that simulation type.

        TODO
        ----
        * Work out how to deal with analytical estimates here
        """
        comp_dGs = []
        solv_dGs = []

        for pus in self.data['solvent'].values():
            solv_dGs.append((
                pus[0].outputs['unit_estimate'],
                pus[0].outputs['unit_estimate_error']
            ))

        for pus in self.data['complex'].values():
            comp_dGs.append((
                pus[0].outputs['unit_estimate'],
                pus[0].outputs['unit_estimate_error']
            ))

        return {'complex': comp_dGs, 'solvent': solv_dGs}

    def get_estimate(self):
        """Get the binding free energy estimate for this calculation.

        Returns
        -------
        dG : unit.Quantity
          The binding free energy. This is a Quantity defined with units.

        TODO
        ----
        * Deal with analytical free enenergy
        """
        def _get_average(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.average(dGs) * u

        individual_estimates = self.get_individual_estimates()
        solv_dG = _get_average(individual_estimates['solvent'])
        comp_dG = _get_average(individual_estimates['complex'])

        return solv_dG - comp_dG

    def get_uncertainty(self):
        """Get the binding free energy error for this calculation.

        Returns
        -------
        err : unit.Quantity
          The standard deviation between estimates of the binding free
          energy. This is a Quantity defined with units.
        """
        def _get_stdev(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.std(dGs) * u

        individual_estimates = self.get_individual_estimates()
        solv_err = _get_stdev(individual_estimates['solvent'])
        comp_err = _get_stdev(individual_estimates['complex'])

        # return the combined error
        return np.sqrt(solv_err**2 + comp_err**2)

    def get_forward_and_reverse_energy_analysis(self) -> dict[str, list[dict[str, Union[npt.NDArray, unit.Quantity]]]]:
        """
        Get the reverse and forward analysis of the free energies.

        Returns
        -------
        forward_reverse : dict[str, list[dict[str, Union[npt.NDArray, unit.Quantity]]]]
            A dictionary, keyed `complex` and `solvent` for each leg of the
            thermodynamic cycle which each contain a list of dictionaries
            containing the forward and reverse analysis of each repeat
            of that simulation type.

            The forward and reverse analysis dictionaries contain:
              - `fractions`: npt.NDArray
                  The fractions of data used for the estimates
              - `forward_DGs`, `reverse_DGs`: unit.Quantity
                  The forward and reverse estimates for each fraction of data
              - `forward_dDGs`, `reverse_dDGs`: unit.Quantity
                  The forward and reverse estimate uncertainty for each
                  fraction of data.
        """

        forward_reverse: dict[str, list[dict[str, Union[npt.NDArray, unit.Quantity]]]] = {}

        for key in ['complex', 'solvent']:
            forward_reverse[key] = [
                pus[0].outputs['forward_and_reverse_energies']
                for pus in self.data[key].values()
            ]

        return forward_reverse

    def get_overlap_matrices(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get a the MBAR overlap estimates for all legs of the simulation.

        Returns
        -------
        overlap_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary with keys `complex` and `solvent` for each
          leg of the thermodynamic cycle, which each containing a
          list of dictionaries with the MBAR overlap estimates of
          each repeat of that simulation type.

          The underlying MBAR dictionaries contain the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats: dict[str, list[dict[str, npt.NDArray]]] = {}

        for key in ['complex', 'solvent']:
            overlap_stats[key] = [
                pus[0].outputs['unit_mbar_overlap']
                for pus in self.data[key].values()
            ]

        return overlap_stats

    def get_replica_transition_statistics(self) -> dict[str, list[dict[str, npt.NDArray]]]:
        """
        Get the replica exchange transition statistics for all
        legs of the simulation.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : dict[str, list[dict[str, npt.NDArray]]]
          A dictionary with keys `complex` and `solvent` for each
          leg of the thermodynamic cycle, which each containing
          a list of dictionaries containing the replica transition
          statistics for each repeat of that simulation type.

          The replica transition statistics dictionaries contain the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        repex_stats: dict[str, list[dict[str, npt.NDArray]]] = {}
        try:
            for key in ['complex', 'solvent']:
                repex_stats[key] = [
                    pus[0].outputs['replica_exchange_statistics']
                    for pus in self.data[key].values()
                ]
        except KeyError:
            errmsg = ("Replica exchange statistics were not found, "
                      "did you run a repex calculation?")
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> dict[str, list[npt.NDArray]]:
        """
        Get the timeseries of replica states for all simulation legs.

        Returns
        -------
        replica_states : dict[str, list[npt.NDArray]]
          Dictionary keyed `complex` and `solvent` for each leg of
          the thermodynamic cycle, with lists of replica states
          timeseries for each repeat of that simulation type.
        """
        replica_states: dict[str, list[npt.NDArray]] = {}

        for key in ['complex', 'solvent']:
            replica_states[key] = [
                pus[0].outputs['replica_states']
                for pus in self.data[key].values()
            ]
        return replica_states

    def equilibration_iterations(self) -> dict[str, list[float]]:
        """
        Get the number of equilibration iterations for each simulation.

        Returns
        -------
        equilibration_lengths : dict[str, list[float]]
          Dictionary keyed `complex` and `solvent` for each leg
          of the thermodynamic cycle, with lists containing the
          number of equilibration iterations for each repeat
          of that simulation type.
        """
        equilibration_lengths: dict[str, list[float]] = {}

        for key in ['complex', 'solvent']:
            equilibration_lengths[key] = [
                pus[0].outputs['equilibration_iterations']
                for pus in self.data[key].values()
            ]

        return equilibration_lengths

    def production_iterations(self) -> dict[str, list[float]]:
        """
        Get the number of production iterations for each simulation.
        Returns the number of uncorrelated production samples for each
        repeat of the calculation.

        Returns
        -------
        production_lengths : dict[str, list[float]]
          Dictionary keyed `complex` and `solvent` for each leg of the
          thermodynamic cycle, with lists with the number
          of production iterations for each repeat of that simulation
          type.
        """
        production_lengths: dict[str, list[float]] = {}

        for key in ['complex', 'solvent']:
            production_lengths[key] = [
                pus[0].outputs['production_iterations']
                for pus in self.data[key].values()
            ]

        return production_lengths


class AbsoluteBindingProtocol(gufe.Protocol):
    """
    Absolute binding free energy calculations using OpenMM and OpenMMTools.

    See Also
    --------
    openfe.protocols
    openfe.protocols.openmm_afe.AbsoluteBindingSettings
    openfe.protocols.openmm_afe.AbsoluteBindingProtocolResult
    openfe.protocols.openmm_afe.AbsoluteBindingComplexUnit
    openfe.protocols.openmm_afe.AbsoluteBindingSolventUnit
    """
    result_cls = AbsoluteBindingProtocolResult
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
          A set of default settings for the Protocol.
        """
        return AbsoluteBindingSettings(
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            complex_system_settings=SystemSettings(),
            solvent_system_settings=SystemSettings(nonbonded_method='nocutoff'),
            alchemical_settings=AlchemicalSettings(),
            alchemsampler_settings=AlchemicalSamplerSettings(n_replicas=24),
            solvation_settings=SolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            complex_simulation_settings=SimulationSettings(
                equilibration_length=2.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
                output_filename='complex.nc',
                checkpoint_storage='complex_checkpoint.nc',
            ),
            solvent_simulation_settings=SimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
                output_filename='solvent.nc',
                checkpoint_storage='solvent_checkpoint.nc',
            ),
        )

    @staticmethod
    def _validate_binding_endstates(
        stateA: ChemicalSystem, stateB: ChemicalSystem,
    ) -> None:
        """
        Initial validation of the endstates for a binding transformation.

        A binding transformation is defined (in terms of gufe components)
        as starting from one or more ligands and a host in solvent
        and ending up in a state with one or more missing ligands.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If there are no SmallMoleculeComponents in stateA
          If there is only one non-Solvent component in stateA
          If there is no SolventComponent in either stateA or stateB
        """
        # Check that there are too few non-SolventComponents in stateA
        non_solv_comps = [i for i in stateA.values()
                          if not isinstance(i, SolventComponent)]

        if not any(
            isinstance(i, SmallMoleculeComponent) for i in non_solv_comps
        ):
            errmsg = "No SmallMoleculeComponent in stateA"
            raise ValueError(errmsg)

        if len(non_solv_comps) < 2:
            errmsg = ("This protocol expects at least two "
                      "non-SolventComponent components in state A "
                      f"Non-SolventComponent entires are: {non_solv_comps}")
            raise ValueError(errmsg)

        # check that there is a solvent component
        if not any(
            isinstance(comp, SolventComponent) for comp in stateA.values()
        ):
            errmsg = "No SolventComponent found in stateA"
            raise ValueError(errmsg)

        if not any(
            isinstance(comp, SolventComponent) for comp in stateB.values()
        ):
            errmsg = "No SolventComponent found in stateB"
            raise ValueError(errmsg)

    @staticmethod
    def _validate_alchemical_components(
        alchemical_components: dict[str, list[Component]]
    ) -> None:
        """
        Checks that the ChemicalSystem alchemical components are correct.

        Parameters
        ----------
        alchemical_components : Dict[str, list[Component]]
          Dictionary containing the alchemical components for
          stateA and stateB.

        Raises
        ------
        ValueError
          If there are alchemical components in state B.
          If there are non SmallMoleculeComponent alchemical species.
          If there are more than one alchemical species.

        Notes
        -----
        * Currently doesn't support alchemical components in state B.
        * Currently doesn't support alchemical components which are not
          SmallMoleculeComponents.
        * Currently doesn't support more than one alchemical component
          being unbound.
        """

        # Crash out if there are any alchemical components in state B for now
        if len(alchemical_components['stateB']) > 0:
            errmsg = ("Components appearing in state B are not "
                      "currently supported")
            raise ValueError(errmsg)

        if len(alchemical_components['stateA']) > 1:
            errmsg = ("More than one alchemical components is not supported "
                      "for absolute binding free energies")
            raise ValueError(errmsg)

        # Crash out if any of the alchemical components are not
        # SmallMoleculeComponent
        for comp in alchemical_components['stateA']:
            if not isinstance(comp, SmallMoleculeComponent):
                errmsg = ("Non SmallMoleculeComponent alchemical species "
                          "are not currently supported")
                raise ValueError(errmsg)

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: extensions
        if extends:  # pragma: no-cover
            raise NotImplementedError("Can't extend simulations yet")

        # Validate components and get alchemical components
        self._validate_binding_endstates(stateA, stateB)
        alchem_comps = system_validation.get_alchemical_components(
            stateA, stateB,
        )
        self._validate_alchemical_components(alchem_comps)

        # Check nonbond & solvent compatibility
        solv_nonbonded_method = self.settings.system_settings.nonbonded_method
        system_validation.validate_solvent(stateA, solv_nonbonded_method)

        # Get the name of the alchemical species
        alchname = alchem_comps['stateA'][0].name

        # Create list units for vacuum and solvent transforms

        solvent_units = [
            AbsoluteBindingSolventUnit(
                # These don't really reflect the actual transform
                # Should these be overriden to be ChemicalSystem{smc} -> ChemicalSystem{} ?
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchem_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Binding, {alchname} solvent leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        complex_units = [
            AbsoluteBindingComplexUnit(
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchem_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Binding, {alchname} complex leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        return solvent_units + complex_units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> Dict[str, Dict[str, Any]]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        unsorted_solvent_repeats = defaultdict(list)
        unsorted_complex_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue
                if pu.outputs['simtype'] == 'solvent':
                    unsorted_solvent_repeats[pu.outputs['repeat_id']].append(pu)
                else:
                    unsorted_complex_repeats[pu.outputs['repeat_id']].append(pu)

        repeats: dict[str, dict[str, list[gufe.ProtocolUnitResult]]] = {
            'solvent': {}, 'complex': {},
        }

        for k, v in unsorted_solvent_repeats.items():
            repeats['solvent'][str(k)] = sorted(
                v, key=lambda x: x.outputs['generation']
            )

        for k, v in unsorted_complex_repeats.items():
            repeats['complex'][str(k)] = sorted(
                v, key=lambda x: x.outputs['generation']
            )

        return repeats


class AbsoluteBindingSolventUnit(BaseAbsoluteUnit):
    def _get_components(self):
        """
        Get the relevant components for a solvent transformation.

        Note
        -----
        The solvent portion of the transformation is the resolvation
        of alchemical species being unbound. The only thing that
        should be present is the alchemical species and the SolventComponent.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        small_mols = {m: m.to_openff()
                      for m in alchem_comps['stateA']}

        solv_comp, _, _ = system_validation.get_components(stateA)

        # 1. We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # 2. ProteinComps can't be alchem_comps (for now?), so will
        # be returned as None
        return alchem_comps, solv_comp, None, small_mols

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a vacuum transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * system_settings : SystemSettings
            * solvation_settings : SolvationSettings
            * alchemical_settings : AlchemicalSettings
            * sampler_settings : AlchemicalSamplerSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * simulation_settings : SimulationSettings
        """
        prot_settings = self._inputs['settings']

        settings = {}
        settings['forcefield_settings'] = prot_settings.forcefield_settings
        settings['thermo_settings'] = prot_settings.thermo_settings
        settings['system_settings'] = prot_settings.system_settings
        settings['solvation_settings'] = prot_settings.solvation_settings
        settings['alchemical_settings'] = prot_settings.alchemical_settings
        settings['sampler_settings'] = prot_settings.alchemsampler_settings
        settings['engine_settings'] = prot_settings.engine_settings
        settings['integrator_settings'] = prot_settings.integrator_settings
        settings['simulation_settings'] = prot_settings.solvent_simulation_settings

        settings_validation.validate_timestep(
            settings['forcefield_settings'].hydrogen_mass,
            settings['integrator_settings'].timestep
        )

        return settings

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> Dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        with without_oechem_backend():
            outputs = self.run(scratch_basepath=ctx.scratch,
                               shared_basepath=ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            'simtype': 'solvent',
            **outputs
        }


class AbsoluteBindingComplexUnit(BaseAbsoluteUnit):
    def _get_components(self):
        """
        Get the relevant components for a complex transformation.

        Returns
        -------
        alchem_comps : dict[str, Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : dict[SmallMoleculeComponent: OFFMolecule]
          SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)
        small_mols = {m: m.to_openff() for m in small_mols}

        return alchem_comps, solv_comp, prot_comp, small_mols

    def _handle_settings(self) -> dict[str, SettingsBaseModel]:
        """
        Extract the relevant settings for a vacuum transformation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : OpenMMSystemGeneratorFFSettings
            * thermo_settings : ThermoSettings
            * system_settings : SystemSettings
            * solvation_settings : SolvationSettings
            * alchemical_settings : AlchemicalSettings
            * sampler_settings : AlchemicalSamplerSettings
            * engine_settings : OpenMMEngineSettings
            * integrator_settings : IntegratorSettings
            * simulation_settings : SimulationSettings
        """
        prot_settings = self._inputs['settings']

        settings = {}
        settings['forcefield_settings'] = prot_settings.forcefield_settings
        settings['thermo_settings'] = prot_settings.thermo_settings
        settings['system_settings'] = prot_settings.system_settings
        settings['solvation_settings'] = prot_settings.solvation_settings
        settings['alchemical_settings'] = prot_settings.alchemical_settings
        settings['sampler_settings'] = prot_settings.alchemsampler_settings
        settings['engine_settings'] = prot_settings.engine_settings
        settings['integrator_settings'] = prot_settings.integrator_settings
        settings['simulation_settings'] = prot_settings.complex_simulation_settings

        settings_validation.validate_timestep(
            settings['forcefield_settings'].hydrogen_mass,
            settings['integrator_settings'].timestep
        )

        return settings

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> Dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        with without_oechem_backend():
            outputs = self.run(scratch_basepath=ctx.scratch,
                               shared_basepath=ctx.shared)

        # TODO: add in the structure checks when they are finished

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            'simtype': 'complex',
            **outputs
        }