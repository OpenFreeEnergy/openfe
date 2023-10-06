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
* Disapearing molecules are only allowed in state A. Support for
  appearing molecules will be added in due course.
* Only small molecules are allowed to act as alchemical molecules.
  Alchemically changing protein or solvent components would induce
  perturbations which are too large to be handled by this Protocol.


Acknowledgements
----------------
* Originally based on hydration.py in
  `espaloma <https://github.com/choderalab/espaloma_charge>`_
"""
from __future__ import annotations

import logging

from collections import defaultdict
import gufe
from gufe.components import Component
import itertools
import numpy as np
from openff.units import unit
from typing import Dict, Optional
from typing import Any, Iterable

from gufe import (
    settings, ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteSolvationSettings, SystemSettings,
    SolvationSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings,
)
from ..openmm_utils import system_validation, settings_validation
from .base import BaseAbsoluteTransformUnit
from openfe.utils import without_oechem_backend, log_system_probe

logger = logging.getLogger(__name__)


class AbsoluteSolvationProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a AbsoluteSolventTransform

    TODO
    ----
    * Add in methods to retreive forward/backwards analyses
    * Add in methods to retreive the overlap matrices
    * Add in method to get replica transition stats
    * Add in method to get replica states
    * Add in method to get equilibration and production iterations
    """
    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    def get_vacuum_individual_estimates(self) -> list[tuple[unit.Quantity, unit.Quantity]]:
        """
        Return a list of tuples containing the individual free energy
        estimates and associated MBAR errors for each repeat of the vacuum
        calculation.

        Returns
        -------
        dGs : list[tuple[unit.Quantity, unit.Quantity]]
        """
        dGs = []

        for pus in self.data.values():
            if pus[0].outputs['simtype'] == 'vacuum':
                dGs.append((
                    pus[0].outputs['unit_estimate'],
                    pus[0].outputs['unit_estimate_error']
                ))

        return dGs

    def get_solvent_individual_estimates(self) -> list[tuple[unit.Quantity, unit.Quantity]]:
        """
        Return a list of tuples containing the individual free energy
        estimates and associated MBAR errors for each repeat of the solvent
        calculation.

        Returns
        -------
        dGs : list[tuple[unit.Quantity, unit.Quantity]]
        """
        dGs = []

        for pus in self.data.values():
            if pus[0].outputs['simtype'] == 'solvent':
                dGs.append((
                    pus[0].outputs['unit_estimate'],
                    pus[0].outputs['unit_estimate_error']
                ))

        return dGs

    def get_estimate(self):
        """Get the solvation free energy estimate for this calculation.

        Returns
        -------
        dG : unit.Quantity
          The solvation free energy. This is a Quantity defined with units.
        """
        def _get_average(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.average(dGs) * u

        vac_dG = _get_average(self.get_vacuum_individual_estimates(self))
        solv_dG = _get_average(self.get_solvent_individual_estimates(self))

        return vac_dG - solv_dG

    def get_uncertainty(self):
        """Get the solvation free energy error for this calculation.

        Returns
        -------
        err : unit.Quantity
          The standard deviation between estimates of the solvation free
          energy. This is a Quantity defined with units.
        """
        def _get_stdev(estimates):
            # Get the unit value of the first value in the estimates
            u = estimates[0][0].u
            # Loop through estimates and get the free energy values
            # in the unit of the first estimate
            dGs = [i[0].to(u).m for i in estimates]

            return np.std(dGs) * u

        vac_err = _get_stdev(self.get_vacuum_individual_estimates(self))
        solv_err = _get_stdev(self.get_solvent_individual_estimates(self))

        # return the combined error
        return np.sqrt(vac_err**2 + solv_err**2)


class AbsoluteSolvationProtocol(gufe.Protocol):
    result_cls = AbsoluteSolvationProtocolResult
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
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            solvent_system_settings=SystemSettings(),
            vacuum_system_settings=SystemSettings(nonbonded_method='nocutoff'),
            alchemical_settings=AlchemicalSettings(),
            alchemsampler_settings=AlchemicalSamplerSettings(
                n_replicas=24,
            ),
            solvation_settings=SolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            solvent_simulation_settings=SimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=10.0 * unit.nanosecond,
                output_filename='solvent.nc',
                checkpoint_storage='solvent_checkpoint.nc',
            ),
            vacuum_simulation_settings=SimulationSettings(
                equilibration_length=0.5 * unit.nanosecond,
                production_length=2.0 * unit.nanosecond,
                output_filename='vacuum.nc',
                checkpoint_storage='vacuum_checkpoint.nc'
            ),
        )

    @staticmethod
    def _validate_solvent_endstates(
        stateA: ChemicalSystem, stateB: ChemicalSystem,
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
          If stateA or stateB contains a ProteinComponent
          If there is no SolventComponent in either stateA or stateB
        """
        # Check that there are no protein components
        for comp in itertools.chain(stateA.values(), stateB.values()):
            if isinstance(comp, ProteinComponent):
                errmsg = ("Protein components are not allowed for "
                          "absolute solvation free energies")
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
          being desolvated.
        """

        # Crash out if there are any alchemical components in state B for now
        if len(alchemical_components['stateB']) > 0:
            errmsg = ("Components appearing in state B are not "
                      "currently supported")
            raise ValueError(errmsg)

        if len(alchemical_components['stateA']) > 1:
            errmsg = ("More than one alchemical components is not supported "
                      "for absolute solvation free energies")
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
        self._validate_solvent_endstates(stateA, stateB)
        alchem_comps = system_validation.get_alchemical_components(
            stateA, stateB,
        )
        self._validate_alchemical_components(alchem_comps)

        # Check nonbond & solvent compatibility
        solv_nonbonded_method = self.settings.solvent_system_settings.nonbonded_method
        vac_nonbonded_method = self.settings.vacuum_system_settings.nonbonded_method
        # Use the more complete system validation solvent checks
        system_validation.validate_solvent(stateA, solv_nonbonded_method)
        # Gas phase is always gas phase
        if vac_nonbonded_method.lower() != 'nocutoff':
            errmsg = ("Only the nocutoff nonbonded_method is supported for "
                      f"vacuum calculations, {vac_nonbonded_method} was "
                      "passed")
            raise ValueError(errmsg)

        # Get the name of the alchemical species
        alchname = alchem_comps['stateA'][0].name

        # Create list units for vacuum and solvent transforms

        solvent_units = [
            AbsoluteSolventTransformUnit(
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchem_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Solvation, {alchname} solvent leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        vacuum_units = [
            AbsoluteVacuumTransformUnit(
                # These don't really reflect the actual transform
                # Should these be overriden to be ChemicalSystem{smc} -> ChemicalSystem{} ?
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchem_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Solvation, {alchname} solvent leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        return solvent_units + vacuum_units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> Dict[str, Any]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue
                unsorted_repeats[pu.outputs['repeat_id']].append(pu)

        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs['generation'])
        return repeats


class AbsoluteVacuumTransformUnit(BaseAbsoluteTransformUnit):
    def _get_components(self):
        """
        Get the relevant components for a vacuum transformation.

        Returns
        -------
        alchem_comps : list[Component]
          A list of alchemical components
        solv_comp : None
          For the gas phase transformation, None will always be returned
          for the solvent component of the chemical system.
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : list[SmallMoleculeComponent]
          A list of SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        _, prot_comp, small_mols = system_validation.get_components(stateA)

        # Note our input state will contain a solvent, we ``None`` that out
        # since this is the gas phase unit.
        return alchem_comps, None, prot_comp, small_mols

    def _handle_settings(self):
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
        settings['system_settings'] = prot_settings.vacuum_system_settings
        settings['solvation_settings'] = prot_settings.solvation_settings
        settings['alchemical_settings'] = prot_settings.alchemical_settings
        settings['sampler_settings'] = prot_settings.alchemsampler_settings
        settings['engine_settings'] = prot_settings.engine_settings
        settings['integrator_settings'] = prot_settings.integrator_settings
        settings['simulation_settings'] = prot_settings.vacuum_simulation_settings

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
            'simtype': 'vacuum',
            **outputs
        }


class AbsoluteSolventTransformUnit(BaseAbsoluteTransformUnit):
    def _get_components(self):
        """
        Get the relevant components for a vacuum transformation.

        Returns
        -------
        alchem_comps : list[Component]
          A list of alchemical components
        solv_comp : SolventComponent
          The SolventComponent of the system
        prot_comp : Optional[ProteinComponent]
          The protein component of the system, if it exists.
        small_mols : list[SmallMoleculeComponent]
          A list of SmallMoleculeComponents to add to the system.
        """
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']

        solv_comp, prot_comp, small_mols = system_validation.get_components(stateA)

        # We don't need to check that solv_comp is not None, otherwise
        # an error will have been raised when calling `validate_solvent`
        # in the Protocol's `_create`.
        # Similarly we don't need to check prot_comp since that's also
        # disallowed on create
        return alchem_comps, solv_comp, prot_comp, small_mols

    def _handle_settings(self):
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
        settings['system_settings'] = prot_settings.solvent_system_settings
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
