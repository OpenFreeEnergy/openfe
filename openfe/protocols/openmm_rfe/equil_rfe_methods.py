# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy methods using OpenMM in a
Perses-like manner.

This module implements the necessary methodology toolking to run calculate a
ligand relative free energy transformation using OpenMM tools and one of the
following methods:
    - Hamiltonian Replica Exchange
    - Self-adjusted mixture sampling
    - Independent window sampling

TODO
----
* Improve this docstring by adding an example use case.

"""
from __future__ import annotations

import os
import logging
from collections import defaultdict
import uuid
import warnings
from itertools import chain
import numpy as np
import numpy.typing as npt
from openff.units import unit
from openff.units.openmm import to_openmm, from_openmm, ensure_quantity
from openff.toolkit.topology import Molecule as OFFMolecule
from openmmtools import multistate
from typing import Optional
from openmm import unit as omm_unit
from openmm.app import PDBFile
import pathlib
from typing import Any, Iterable, Union
import openmmtools
import mdtraj

import gufe
from gufe import (
    settings, ChemicalSystem, LigandAtomMapping, Component, ComponentMapping,
    SmallMoleculeComponent, ProteinComponent,
)

from .equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings, SystemSettings,
    SolvationSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings
)
from ..openmm_utils import (
    system_validation, settings_validation, system_creation,
    multistate_analysis
)
from . import _rfe_utils
from ...utils import without_oechem_backend, log_system_probe

logger = logging.getLogger(__name__)


def _get_resname(off_mol) -> str:
    # behaviour changed between 0.10 and 0.11
    omm_top = off_mol.to_topology().to_openmm()
    names = [r.name for r in omm_top.residues()]
    if len(names) > 1:
        raise ValueError("We assume single residue")
    return names[0]


def _validate_alchemical_components(
        alchemical_components: dict[str, list[Component]],
        mapping: Optional[dict[str, ComponentMapping]],
):
    """
    Checks that the alchemical components are suitable for the RFE protocol.

    Specifically we check:
      1. That all alchemical components are mapped.
      2. That all alchemical components are SmallMoleculeComponents.
      3. If the mappings involves element changes in core atoms

    Parameters
    ----------
    alchemical_components : dict[str, list[Component]]
      Dictionary contatining the alchemical components for
      states A and B.
    mapping : dict[str, ComponentMapping]
      Dictionary of mappings between transforming components.

    Raises
    ------
    ValueError
      * If there are more than one mapping or mapping is None
      * If there are any unmapped alchemical components.
      * If there are any alchemical components that are not
        SmallMoleculeComponents.
    UserWarning
      * Mappings which involve element changes in core atoms
    """
    # Check mapping
    # For now we only allow for a single mapping, this will likely change
    if mapping is None or len(mapping.values()) > 1:
        errmsg = "A single LigandAtomMapping is expected for this Protocol"
        raise ValueError(errmsg)

    # Check that all alchemical components are mapped & small molecules
    mapped = {}
    mapped['stateA'] = [m.componentA for m in mapping.values()]
    mapped['stateB'] = [m.componentB for m in mapping.values()]

    for idx in ['stateA', 'stateB']:
        if len(alchemical_components[idx]) != len(mapped[idx]):
            errmsg = f"missing alchemical components in {idx}"
            raise ValueError(errmsg)
        for comp in alchemical_components[idx]:
            if comp not in mapped[idx]:
                raise ValueError(f"Unmapped alchemical component {comp}")
            if not isinstance(comp, SmallMoleculeComponent):  # pragma: no-cover
                errmsg = ("Transformations involving non "
                          "SmallMoleculeComponent species {comp} "
                          "are not currently supported")
                raise ValueError(errmsg)

    # Validate element changes in mappings
    for m in mapping.values():
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
                    "simulation")
                logger.warn(wmsg)
                warnings.warn(wmsg)  # TODO: remove this once logging is fixed


class RelativeHybridTopologyProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a RelativeHybridTopologyProtocol"""
    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    def get_estimate(self) -> unit.Quantity:
        """Average free energy difference of this transformation

        Returns
        -------
        dG : unit.Quantity
          The free energy difference between the first and last states. This is
          a Quantity defined with units.
        """
        # TODO: Check this holds up completely for SAMS.
        dGs = [pus[0].outputs['unit_estimate'] for pus in self.data.values()]
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = [dG.to(u).m for dG in dGs]

        return np.average(vals) * u

    def get_uncertainty(self) -> unit.Quantity:
        """The uncertainty/error in the dG value: The std of the estimates of
        each independent repeat
        """
        dGs = [pus[0].outputs['unit_estimate'] for pus in self.data.values()]
        u = dGs[0].u
        # convert all values to units of the first value, then take average of magnitude
        # this would avoid a screwy case where each value was in different units
        vals = [dG.to(u).m for dG in dGs]

        return np.std(vals) * u

    def get_individual_estimates(self) -> list[tuple[unit.Quantity, unit.Quantity]]:
        """Return a list of tuples containing the individual free energy
        estimates and associated MBAR errors for each repeat.

        Returns
        -------
        dGs : list[tuple[unit.Quantity]]
          n_replicate simulation list of tuples containing the free energy
          estimates (first entry) and associated MBAR estimate errors
          (second entry).
        """
        dGs = [(pus[0].outputs['unit_estimate'],
                pus[0].outputs['unit_estimate_error'])
               for pus in self.data.values()]
        return dGs

    def get_forward_and_reverse_energy_analysis(self) -> list[dict[str, Union[npt.NDArray, unit.Quantity]]]:
        """
        Get a list of forward and reverse analysis of the free energies
        for each repeat using uncorrelated production samples.

        The returned dicts have keys:
        'fractions' - the fraction of data used for this estimate
        'forward_DGs', 'reverse_DGs' - for each fraction of data, the estimate
        'forward_dDGs', 'reverse_dDGs' - for each estimate, the uncertainty

        The 'fractions' values are a numpy array, while the other arrays are
        Quantity arrays, with units attached.

        Returns
        -------
        forward_reverse : dict[str, Union[npt.NDArray, unit.Quantity]]
        """
        forward_reverse = [pus[0].outputs['forward_and_reverse_energies']
                           for pus in self.data.values()]

        return forward_reverse

    def get_overlap_matrices(self) -> list[dict[str, npt.NDArray]]:
        """
        Return a list of dictionary containing the MBAR overlap estimates
        calculated for each repeat.

        Returns
        -------
        overlap_stats : list[dict[str, npt.NDArray]]
          A list of dictionaries containing the following keys:
            * ``scalar``: One minus the largest nontrivial eigenvalue
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              overlap matrix
            * ``matrix``: Estimated overlap matrix of observing a sample from
              state i in state j
        """
        # Loop through and get the repeats and get the matrices
        overlap_stats = [pus[0].outputs['unit_mbar_overlap']
                         for pus in self.data.values()]

        return overlap_stats

    def get_replica_transition_statistics(self) -> list[dict[str, npt.NDArray]]:
        """The replica lambda state transition statistics for each repeat.

        Note
        ----
        This is currently only available in cases where a replica exchange
        simulation was run.

        Returns
        -------
        repex_stats : list[dict[str, npt.NDArray]]
          A list of dictionaries containing the following:
            * ``eigenvalues``: The sorted (descending) eigenvalues of the
              lambda state transition matrix
            * ``matrix``: The transition matrix estimate of a replica switching
              from state i to state j.
        """
        try:
            repex_stats = [pus[0].outputs['replica_exchange_statistics']
                           for pus in self.data.values()]
        except KeyError:
            errmsg = ("Replica exchange statistics were not found, "
                      "did you run a repex calculation?")
            raise ValueError(errmsg)

        return repex_stats

    def get_replica_states(self) -> list[npt.NDArray]:
        """
        Returns the timeseries of replica states for each repeat.

        Returns
        -------
        replica_states : List[npt.NDArray]
          List of replica states for each repeat
        """
        replica_states = [pus[0].outputs['replica_states']
                          for pus in self.data.values()]

        return replica_states

    def equilibration_iterations(self) -> list[float]:
        """
        Returns the number of equilibration iterations for each repeat
        of the calculation.

        Returns
        -------
        equilibration_lengths : list[float]
        """
        equilibration_lengths = [pus[0].outputs['equilibration_iterations']
                                 for pus in self.data.values()]

        return equilibration_lengths

    def production_iterations(self) -> list[float]:
        """
        Returns the number of uncorrelated production samples for each
        repeat of the calculation.

        Returns
        -------
        production_lengths : list[float]
        """
        production_lengths = [pus[0].outputs['production_iterations']
                              for pus in self.data.values()]

        return production_lengths


class RelativeHybridTopologyProtocol(gufe.Protocol):
    result_cls = RelativeHybridTopologyProtocolResult
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
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            system_settings=SystemSettings(),
            solvation_settings=SolvationSettings(),
            alchemical_settings=AlchemicalSettings(),
            alchemical_sampler_settings=AlchemicalSamplerSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            simulation_settings=SimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            )
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, gufe.ComponentMapping]] = None,
        extends: Optional[gufe.ProtocolDAGResult] = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: Extensions?
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        # Get alchemical components & validate them + mapping
        alchem_comps = system_validation.get_alchemical_components(
            stateA, stateB
        )
        _validate_alchemical_components(alchem_comps, mapping)

        # For now we've made it fail already if it was None,
        ligandmapping = list(mapping.values())[0]  # type: ignore

        # Validate solvent component
        nonbond = self.settings.system_settings.nonbonded_method
        solv_comp = system_validation.validate_solvent(stateA, nonbond)

        # make sure that the solvation backend is correct
        settings_validation.validate_solvent_settings(
            self.settings.solvation_settings,
            solv_comp,
            allowed_backends=['openmm']
        )

        if self.settings.solvation_settings.backend.lower() != 'openmm':
            errmsg = "non openmm solvation backend is not supported"
            raise ValueError(errmsg)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
        Anames = ','.join(c.name for c in alchem_comps['stateA'])
        Bnames = ','.join(c.name for c in alchem_comps['stateB'])
        # our DAG has no dependencies, so just list units
        n_repeats = self.settings.alchemical_sampler_settings.n_repeats
        units = [RelativeHybridTopologyProtocolUnit(
            stateA=stateA, stateB=stateB, ligandmapping=ligandmapping,
            settings=self.settings,
            generation=0, repeat_id=int(uuid.uuid4()),
            name=f'{Anames} to {Bnames} repeat {i} generation 0')
            for i in range(n_repeats)]

        return units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, Any]:
        # result units will have a repeat_id and generations within this repeat_id
        # first group according to repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue

                unsorted_repeats[pu.outputs['repeat_id']].append(pu)

        # then sort by generation within each repeat_id list
        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs['generation'])

        # returns a dict of repeat_id: sorted list of ProtocolUnitResult
        return repeats


class RelativeHybridTopologyProtocolUnit(gufe.ProtocolUnit):
    """
    Calculates the relative free energy of an alchemical ligand transformation.
    """

    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 ligandmapping: LigandAtomMapping,
                 settings: RelativeHybridTopologyProtocolSettings,
                 generation: int,
                 repeat_id: int,
                 name: Optional[str] = None,
                 ):
        """
        Parameters
        ----------
        stateA, stateB : ChemicalSystem
          the two ligand SmallMoleculeComponents to transform between.  The
          transformation will go from ligandA to ligandB.
        ligandmapping : LigandAtomMapping
          the mapping of atoms between the two ligand components
        settings : settings.Settings
          the settings for the Method.  This can be constructed using the
          get_default_settings classmethod to give a starting point that
          can be updated to suit.
        repeat_id : int
          identifier for which repeat (aka replica/clone) this Unit is
        generation : int
          counter for how many times this repeat has been extended
        name : str, optional
          human-readable identifier for this Unit

        Notes
        -----
        The mapping used must not involve any elemental changes.  A check for
        this is done on class creation.
        """
        super().__init__(
            name=name,
            stateA=stateA,
            stateB=stateB,
            ligandmapping=ligandmapping,
            settings=settings,
            repeat_id=repeat_id,
            generation=generation
        )

    def run(self, *, dry=False, verbose=True,
            scratch_basepath=None,
            shared_basepath=None) -> dict[str, Any]:
        """Run the relative free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary hybrid
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        scratch_basepath: Pathlike, optional
          Where to store temporary files, defaults to current working directory
        shared_basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.

        Raises
        ------
        error
          Exception if anything failed
        """
        if verbose:
            self.logger.info("Preparing the hybrid topology simulation")
        if scratch_basepath is None:
            scratch_basepath = pathlib.Path('.')
        if shared_basepath is None:
            # use cwd
            shared_basepath = pathlib.Path('.')

        # 0. General setup and settings dependency resolution step

        # Extract relevant settings
        protocol_settings: RelativeHybridTopologyProtocolSettings = self._inputs['settings']
        stateA = self._inputs['stateA']
        stateB = self._inputs['stateB']
        mapping = self._inputs['ligandmapping']

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = protocol_settings.forcefield_settings
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        alchem_settings: AlchemicalSettings = protocol_settings.alchemical_settings
        system_settings: SystemSettings = protocol_settings.system_settings
        solvation_settings: SolvationSettings = protocol_settings.solvation_settings
        sampler_settings: AlchemicalSamplerSettings = protocol_settings.alchemical_sampler_settings
        sim_settings: SimulationSettings = protocol_settings.simulation_settings
        timestep = protocol_settings.integrator_settings.timestep
        mc_steps = protocol_settings.integrator_settings.n_steps.m

        # is the timestep good for the mass?
        settings_validation.validate_timestep(
            forcefield_settings.hydrogen_mass, timestep
        )

        # get the simulation steps
        equil_steps, prod_steps = settings_validation.get_simsteps(
            equil_length=sim_settings.equilibration_length,
            prod_length=sim_settings.production_length,
            timestep=timestep, mc_steps=mc_steps
        )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)

        # 1. Create stateA system
        # a. get a system generator
        if sim_settings.forcefield_cache is not None:
            ffcache = shared_basepath / sim_settings.forcefield_cache
        else:
            ffcache = None

        system_generator = system_creation.get_system_generator(
            forcefield_settings=forcefield_settings,
            thermo_settings=thermo_settings,
            system_settings=system_settings,
            cache=ffcache,
            has_solvent=solvent_comp is not None,
        )

        # workaround for conformer generation failures
        # see openfe issue #576
        # calculate partial charges manually if not already given
        # convert to OpenFF here,
        # and keep the molecule around to maintain the partial charges
        off_small_mols: dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]]
        off_small_mols = {
            'stateA': [(mapping.componentA, mapping.componentA.to_openff())],
            'stateB': [(mapping.componentB, mapping.componentB.to_openff())],
            'both': [(m, m.to_openff()) for m in small_mols
                     if (m != mapping.componentA and m != mapping.componentB)]
        }

        # b. force the creation of parameters
        # This is necessary because we need to have the FF generated ahead of
        # solvating the system.
        # Note: by default this is cached to ctx.shared/db.json so shouldn't
        # incur too large a cost
        self.logger.info("Parameterizing molecules")
        for smc, mol in chain(off_small_mols['stateA'],
                              off_small_mols['stateB'],
                              off_small_mols['both']):
            # robustly calculate partial charges;
            if mol.partial_charges is not None and np.any(mol.partial_charges):
                # skip if we have existing partial charges unless they are zero (see openmmforcefields)
                continue
            try:
                # try and follow official spec method
                mol.assign_partial_charges('am1bcc')
            except ValueError:  # this is what a confgen failure yields
                # but fallback to using existing conformer
                mol.assign_partial_charges('am1bcc',
                                           use_conformers=mol.conformers)

            system_generator.create_system(mol.to_topology().to_openmm(),
                                           molecules=[mol])

        # c. get OpenMM Modeller + a dictionary of resids for each component
        stateA_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_comp,
            solvent_comp=solvent_comp,
            small_mols=dict(chain(off_small_mols['stateA'],
                                  off_small_mols['both'])),
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        # d. get topology & positions
        # Note: roundtrip positions to remove vec3 issues
        stateA_topology = stateA_modeller.getTopology()
        stateA_positions = to_openmm(
            from_openmm(stateA_modeller.getPositions())
        )

        # e. create the stateA System
        stateA_system = system_generator.create_system(
            stateA_modeller.topology,
            molecules=[m for _, m in chain(off_small_mols['stateA'],
                                           off_small_mols['both'])],
        )

        # 2. Get stateB system
        # a. get the topology
        stateB_topology, stateB_alchem_resids = _rfe_utils.topologyhelpers.combined_topology(
            stateA_topology,
            # zeroth item (there's only one) then get the OFF representation
            off_small_mols['stateB'][0][1].to_topology().to_openmm(),
            exclude_resids=comp_resids[mapping.componentA],
        )

        # b. get a list of small molecules for stateB
        stateB_system = system_generator.create_system(
            stateB_topology,
            molecules=[m for _, m in chain(off_small_mols['stateB'],
                                           off_small_mols['both'])],
        )

        #  c. Define correspondence mappings between the two systems
        ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            stateA_system, stateA_topology, comp_resids[mapping.componentA],
            stateB_system, stateB_topology, stateB_alchem_resids,
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        #  d. Finally get the positions
        stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings, stateA_topology, stateB_topology,
            old_positions=ensure_quantity(stateA_positions, 'openmm'),
            insert_positions=ensure_quantity(off_small_mols['stateB'][0][1].conformers[0], 'openmm'),
        )

        # 3. Create the hybrid topology
        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
            stateA_system, stateA_positions, stateA_topology,
            stateB_system, stateB_positions, stateB_topology,
            old_to_new_atom_map=ligand_mappings['old_to_new_atom_map'],
            old_to_new_core_atom_map=ligand_mappings['old_to_new_core_atom_map'],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=alchem_settings.softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchem_settings.interpolate_old_and_new_14s,
            flatten_torsions=alchem_settings.flatten_torsions,
        )

        # 4. Create lambda schedule
        # TODO - this should be exposed to users, maybe we should offer the
        # ability to print the schedule directly in settings?
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=alchem_settings.lambda_functions,
            windows=alchem_settings.lambda_windows
        )

        # PR #125 temporarily pin lambda schedule spacing to n_replicas
        n_replicas = sampler_settings.n_replicas
        if n_replicas != len(lambdas.lambda_schedule):
            errmsg = (f"Number of replicas {n_replicas} "
                      f"does not equal the number of lambda windows "
                      f"{len(lambdas.lambda_schedule)}")
            raise ValueError(errmsg)

        # 9. Create the multistate reporter
        # Get the sub selection of the system to print coords for
        selection_indices = hybrid_factory.hybrid_topology.select(
                sim_settings.output_indices
        )

        #  a. Create the multistate reporter
        nc = shared_basepath / sim_settings.output_filename
        chk = shared_basepath / sim_settings.checkpoint_storage
        reporter = multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=sim_settings.checkpoint_interval.m,
            checkpoint_storage=chk,
        )

        #  b. Write out a PDB containing the subsampled hybrid state
        bfactors = np.zeros_like(selection_indices, dtype=float)  # solvent
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_old_atoms']))] = 0.25  # lig A
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['core_atoms']))] = 0.50  # core
        bfactors[np.in1d(selection_indices, list(hybrid_factory._atom_classes['unique_new_atoms']))] = 0.75  # lig B
        # bfactors[np.in1d(selection_indices, protein)] = 1.0  # prot+cofactor

        if len(selection_indices) > 0:
            traj = mdtraj.Trajectory(
                    hybrid_factory.hybrid_positions[selection_indices, :],
                    hybrid_factory.hybrid_topology.subset(selection_indices),
            ).save_pdb(
                shared_basepath / sim_settings.output_structure,
                bfactors=bfactors,
            )

        # 10. Get platform
        platform = _rfe_utils.compute.get_openmm_platform(
            protocol_settings.engine_settings.compute_platform
        )

        # 11. Set the integrator
        #  a. get integrator settings
        integrator_settings = protocol_settings.integrator_settings

        # Validate settings
        # Virtual sites sanity check - ensure we restart velocities when
        # there are virtual sites in the system
        if hybrid_factory.has_virtual_sites:
            if not integrator_settings.reassign_velocities:
                errmsg = ("Simulations with virtual sites without velocity "
                          "reassignments are unstable in openmmtools")
                raise ValueError(errmsg)

        #  b. create langevin integrator
        integrator = openmmtools.mcmc.LangevinDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.collision_rate),
            n_steps=integrator_settings.n_steps.m,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        # 12. Create sampler
        self.logger.info("Creating and setting up the sampler")
        if sampler_settings.sampler_method.lower() == "repex":
            sampler = _rfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = _rfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations,
                flatness_criteria=sampler_settings.flatness_criteria,
                gamma0=sampler_settings.gamma0,
            )
        elif sampler_settings.sampler_method.lower() == 'independent':
            sampler = _rfe_utils.multistate.HybridMultiStateSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )

        else:
            raise AttributeError(f"Unknown sampler {sampler_settings.sampler_method}")

        sampler.setup(
            n_replicas=sampler_settings.n_replicas,
            reporter=reporter,
            lambda_protocol=lambdas,
            temperature=to_openmm(thermo_settings.temperature),
            endstates=alchem_settings.unsampled_endstates,
            minimization_platform=platform.getName(),
        )

        try:
            # Create context caches (energy + sampler)
            energy_context_cache = openmmtools.cache.ContextCache(
                capacity=None, time_to_live=None, platform=platform,
            )

            sampler_context_cache = openmmtools.cache.ContextCache(
                capacity=None, time_to_live=None, platform=platform,
            )

            sampler.energy_context_cache = energy_context_cache
            sampler.sampler_context_cache = sampler_context_cache

            if not dry:  # pragma: no-cover
                # minimize
                if verbose:
                    self.logger.info("Running minimization")

                sampler.minimize(max_iterations=sim_settings.minimization_steps)

                # equilibrate
                if verbose:
                    self.logger.info("Running equilibration phase")

                sampler.equilibrate(int(equil_steps / mc_steps))  # type: ignore

                # production
                if verbose:
                    self.logger.info("Running production phase")

                sampler.extend(int(prod_steps / mc_steps))  # type: ignore

                self.logger.info("Production phase complete")

                self.logger.info("Post-simulation analysis of results")
                # calculate relevant analyses of the free energies & sampling
                # First close & reload the reporter to avoid netcdf clashes
                analyzer = multistate_analysis.MultistateEquilFEAnalysis(
                    reporter,
                    sampling_method=sampler_settings.sampler_method.lower(),
                    result_units=unit.kilocalorie_per_mole,
                )
                analyzer.plot(filepath=shared_basepath, filename_prefix="")
                analyzer.close()

            else:
                # clean up the reporter file
                fns = [shared_basepath / sim_settings.output_filename,
                       shared_basepath / sim_settings.checkpoint_storage]
                for fn in fns:
                    os.remove(fn)
        finally:
            # close reporter when you're done, prevent
            # file handle clashes
            reporter.close()

            # clear GPU contexts
            # TODO: use cache.empty() calls when openmmtools #690 is resolved
            # replace with above
            for context in list(energy_context_cache._lru._data.keys()):
                del energy_context_cache._lru._data[context]
            for context in list(sampler_context_cache._lru._data.keys()):
                del sampler_context_cache._lru._data[context]
            # cautiously clear out the global context cache too
            for context in list(
                    openmmtools.cache.global_context_cache._lru._data.keys()):
                del openmmtools.cache.global_context_cache._lru._data[context]

            del sampler_context_cache, energy_context_cache

            if not dry:
                del integrator, sampler

        if not dry:  # pragma: no-cover
            return {
                'nc': nc,
                'last_checkpoint': chk,
                **analyzer.unit_results_dict
            }
        else:
            return {'debug': {'sampler': sampler}}

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])
        with without_oechem_backend():
            outputs = self.run(scratch_basepath=ctx.scratch,
                               shared_basepath=ctx.shared)


        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs
        }
