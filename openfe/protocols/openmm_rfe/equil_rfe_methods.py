# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium Relative Free Energy methods using OpenMM and OpenMMTools in a
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

Acknowledgements
----------------
This Protocol is based on, and leverages components originating from
the Perses toolkit (https://github.com/choderalab/perses).
"""
from __future__ import annotations

import os
import logging
from collections import defaultdict
import uuid
import warnings
import json
from itertools import chain
import matplotlib.pyplot as plt
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
import subprocess
from rdkit import Chem

import gufe
from gufe import (
    settings, ChemicalSystem, LigandAtomMapping, Component, ComponentMapping,
    SmallMoleculeComponent, ProteinComponent, SolventComponent,
)

from .equil_rfe_settings import (
    RelativeHybridTopologyProtocolSettings,
    OpenMMSolvationSettings, AlchemicalSettings, LambdaSettings,
    MultiStateSimulationSettings, OpenMMEngineSettings,
    IntegratorSettings, MultiStateOutputSettings,
    OpenFFPartialChargeSettings,
)
from openfe.protocols.openmm_utils.omm_settings import (
    BasePartialChargeSettings,
)
from ..openmm_utils import (
    system_validation, settings_validation, system_creation,
    multistate_analysis, charge_generation
)
from . import _rfe_utils
from ...utils import without_oechem_backend, log_system_probe
from ...analysis import plotting
from openfe.due import due, Doi


logger = logging.getLogger(__name__)


due.cite(Doi("10.5281/zenodo.1297683"),
         description="Perses",
         path="openfe.protocols.openmm_rfe.equil_rfe_methods",
         cite_module=True)

due.cite(Doi("10.5281/zenodo.596622"),
         description="OpenMMTools",
         path="openfe.protocols.openmm_rfe.equil_rfe_methods",
         cite_module=True)

due.cite(Doi("10.1371/journal.pcbi.1005659"),
         description="OpenMM",
         path="openfe.protocols.openmm_rfe.equil_rfe_methods",
         cite_module=True)


def _get_resname(off_mol) -> str:
    # behaviour changed between 0.10 and 0.11
    omm_top = off_mol.to_topology().to_openmm()
    names = [r.name for r in omm_top.residues()]
    if len(names) > 1:
        raise ValueError("We assume single residue")
    return names[0]


def _get_alchemical_charge_difference(
    mapping: LigandAtomMapping,
    nonbonded_method: str,
    explicit_charge_correction: bool,
    solvent_component: SolventComponent
) -> int:
    """
    Checks and returns the difference in formal charge between state A and B.

    Raises
    ------
    ValueError
      * If an explicit charge correction is attempted and the
        nonbonded method is not PME.
      * If the absolute charge difference is greater than one
        and an explicit charge correction is attempted.
    UserWarning
      If there is any charge difference.

    Parameters
    ----------
    mapping : dict[str, ComponentMapping]
      Dictionary of mappings between transforming components.
    nonbonded_method : str
      The OpenMM nonbonded method used for the simulation.
    explicit_charge_correction : bool
      Whether or not to use an explicit charge correction.
    solvent_component : openfe.SolventComponent
      The SolventComponent of the simulation.

    Returns
    -------
    int
      The formal charge difference between states A and B.
      This is defined as sum(charge state A) - sum(charge state B)
    """
    chg_A = Chem.rdmolops.GetFormalCharge(
        mapping.componentA.to_rdkit()
    )
    chg_B = Chem.rdmolops.GetFormalCharge(
        mapping.componentB.to_rdkit()
    )

    difference = chg_A - chg_B

    if abs(difference) > 0:
        if explicit_charge_correction:
            if nonbonded_method.lower() != "pme":
                errmsg = ("Explicit charge correction when not using PME is "
                          "not currently supported.")
                raise ValueError(errmsg)
            if abs(difference) > 1:
                errmsg = (f"A charge difference of {difference} is observed "
                          "between the end states and an explicit charge  "
                          "correction has been requested. Unfortunately "
                          "only absolute differences of 1 are supported.")
                raise ValueError(errmsg)

            ion = {-1: solvent_component.positive_ion,
                   1: solvent_component.negative_ion}[difference]
            wmsg = (f"A charge difference of {difference} is observed "
                    "between the end states. This will be addressed by "
                    f"transforming a water into a {ion} ion")
            logger.warning(wmsg)
            warnings.warn(wmsg)
        else:
            wmsg = (f"A charge difference of {difference} is observed "
                    "between the end states. No charge correction has "
                    "been requested, please account for this in your "
                    "final results.")
            logger.warning(wmsg)
            warnings.warn(wmsg)

    return difference


def _validate_alchemical_components(
    alchemical_components: dict[str, list[Component]],
    mapping: Optional[Union[ComponentMapping, list[ComponentMapping]]],
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
    mapping : Optional[Union[ComponentMapping, list[ComponentMapping]]]
      all mappings between transforming components.

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
    if isinstance(mapping, ComponentMapping):
        mapping = [mapping]
    # Check mapping
    # For now we only allow for a single mapping, this will likely change
    if mapping is None or len(mapping) != 1:
        errmsg = "A single LigandAtomMapping is expected for this Protocol"
        raise ValueError(errmsg)

    # Check that all alchemical components are mapped & small molecules
    mapped = {'stateA': [m.componentA for m in mapping],
              'stateB': [m.componentB for m in mapping]}

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
                    "simulation")
                logger.warning(wmsg)
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
        each independent repeat or the MBAR estimate error for a single repeat
        """
        if len(self.data.values()) > 1:
            dGs = [
                pus[0].outputs['unit_estimate'] for pus in self.data.values()
            ]
            u = dGs[0].u
            # convert all values to units of the first value, then take 
            # average of magnitude. this would avoid a screwy case where each
            # value was in different units
            vals = [dG.to(u).m for dG in dGs]
            unc = np.std(vals) * u
        else:
            # use MBAR estimate error directly for a single repeat
            uncs = [
                pus[0].outputs['unit_estimate_error'] 
                for pus in self.data.values()
            ]
            assert len(uncs) == 1, "Protocols and number of errors mismatch"
            unc = uncs[0]
        return unc

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

    def get_forward_and_reverse_energy_analysis(self) -> list[Optional[dict[str, Union[npt.NDArray, unit.Quantity]]]]:
        """
        Get a list of forward and reverse analysis of the free energies
        for each repeat using uncorrelated production samples.

        The returned dicts have keys:
        'fractions' - the fraction of data used for this estimate
        'forward_DGs', 'reverse_DGs' - for each fraction of data, the estimate
        'forward_dDGs', 'reverse_dDGs' - for each estimate, the uncertainty

        The 'fractions' values are a numpy array, while the other arrays are
        Quantity arrays, with units attached.

        If the list entry is ``None`` instead of a dictionary, this indicates
        that the analysis could not be carried out for that repeat. This
        is most likely caused by MBAR convergence issues when attempting to
        calculate free energies from too few samples.


        Returns
        -------
        forward_reverse : list[Optional[dict[str, Union[npt.NDArray, unit.Quantity]]]]


        Raises
        ------
        UserWarning
          If any of the forward and reverse entries are ``None``.
        """
        forward_reverse = [pus[0].outputs['forward_and_reverse_energies']
                           for pus in self.data.values()]

        if None in forward_reverse:
            wmsg = (
                "One or more ``None`` entries were found in the list of "
                "forward and reverse analyses. This is likely caused by "
                "an MBAR convergence failure caused by too few independent "
                "samples when calculating the free energies of the 10% "
                "timeseries slice."
            )
            warnings.warn(wmsg)

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
        def is_file(filename: str):
            p = pathlib.Path(filename)
            if not p.exists():
                errmsg = f"File could not be found {p}"
                raise ValueError(errmsg)
            return p

        replica_states = []

        for pus in self.data.values():
            nc = is_file(pus[0].outputs['nc'])
            dir_path = nc.parents[0]
            chk = is_file(dir_path / pus[0].outputs['last_checkpoint']).name
            reporter = multistate.MultiStateReporter(
                storage=nc, checkpoint_storage=chk, open_mode='r'
            )
            replica_states.append(
                np.asarray(reporter.read_replica_thermodynamic_states())
            )
            reporter.close()

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
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=OpenMMSolvationSettings(),
            alchemical_settings=AlchemicalSettings(softcore_LJ='gapsys'),
            lambda_settings=LambdaSettings(),
            simulation_settings=MultiStateSimulationSettings(
                equilibration_length=1.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            ),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            output_settings=MultiStateOutputSettings(),
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[Union[gufe.ComponentMapping, list[gufe.ComponentMapping]]],
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
        ligandmapping = mapping[0] if isinstance(mapping, list) else mapping  # type: ignore

        # Validate solvent component
        nonbond = self.settings.forcefield_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
        Anames = ','.join(c.name for c in alchem_comps['stateA'])
        Bnames = ','.join(c.name for c in alchem_comps['stateB'])
        # our DAG has no dependencies, so just list units
        n_repeats = self.settings.protocol_repeats
        units = [RelativeHybridTopologyProtocolUnit(
            protocol=self,
            stateA=stateA, stateB=stateB,
            ligandmapping=ligandmapping,  # type: ignore
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

    def __init__(
        self,
        *,
        protocol: RelativeHybridTopologyProtocol,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        ligandmapping: LigandAtomMapping,
        generation: int,
        repeat_id: int,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        protocol : RelativeHybridTopologyProtocol
          protocol used to create this Unit. Contains key information such
          as the settings.
        stateA, stateB : ChemicalSystem
          the two ligand SmallMoleculeComponents to transform between.  The
          transformation will go from ligandA to ligandB.
        ligandmapping : LigandAtomMapping
          the mapping of atoms between the two ligand components
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
            protocol=protocol,
            stateA=stateA,
            stateB=stateB,
            ligandmapping=ligandmapping,
            repeat_id=repeat_id,
            generation=generation
        )

    @staticmethod
    def _assign_partial_charges(
        charge_settings: OpenFFPartialChargeSettings,
        off_small_mols: dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]],
    ) -> None:
        """
        Assign partial charges to SMCs.

        Parameters
        ----------
        charge_settings : OpenFFPartialChargeSettings
          Settings for controlling how the partial charges are assigned.
        off_small_mols : dict[str, list[tuple[SmallMoleculeComponent, OFFMolecule]]]
          Dictionary of dictionary of OpenFF Molecules to add, keyed by
          state and SmallMoleculeComponent.
        """
        for smc, mol in chain(off_small_mols['stateA'],
                              off_small_mols['stateB'],
                              off_small_mols['both']):
            charge_generation.assign_offmol_partial_charges(
                offmol=mol,
                overwrite=False,
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
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
        protocol_settings: RelativeHybridTopologyProtocolSettings = self._inputs['protocol'].settings
        stateA = self._inputs['stateA']
        stateB = self._inputs['stateB']
        mapping = self._inputs['ligandmapping']

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = protocol_settings.forcefield_settings
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        alchem_settings: AlchemicalSettings = protocol_settings.alchemical_settings
        lambda_settings: LambdaSettings = protocol_settings.lambda_settings
        charge_settings: BasePartialChargeSettings = protocol_settings.partial_charge_settings
        solvation_settings: OpenMMSolvationSettings = protocol_settings.solvation_settings
        sampler_settings: MultiStateSimulationSettings = protocol_settings.simulation_settings
        output_settings: MultiStateOutputSettings = protocol_settings.output_settings
        integrator_settings: IntegratorSettings = protocol_settings.integrator_settings

        # is the timestep good for the mass?
        settings_validation.validate_timestep(
            forcefield_settings.hydrogen_mass,
            integrator_settings.timestep
        )
        # TODO: Also validate various conversions?
        # Convert various time based inputs to steps/iterations
        steps_per_iteration = settings_validation.convert_steps_per_iteration(
            simulation_settings=sampler_settings,
            integrator_settings=integrator_settings,
        )

        equil_steps = settings_validation.get_simsteps(
            sim_length=sampler_settings.equilibration_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )
        prod_steps = settings_validation.get_simsteps(
            sim_length=sampler_settings.production_length,
            timestep=integrator_settings.timestep,
            mc_steps=steps_per_iteration,
        )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(stateA)

        # Get the change difference between the end states
        # and check if the charge correction used is appropriate
        charge_difference = _get_alchemical_charge_difference(
            mapping,
            forcefield_settings.nonbonded_method,
            alchem_settings.explicit_charge_correction,
            solvent_comp,
        )

        # 1. Create stateA system
        self.logger.info("Parameterizing molecules")

        # a. create offmol dictionaries and assign partial charges
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

        self._assign_partial_charges(charge_settings, off_small_mols)

        # b. get a system generator
        if output_settings.forcefield_cache is not None:
            ffcache = shared_basepath / output_settings.forcefield_cache
        else:
            ffcache = None

        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
            system_generator = system_creation.get_system_generator(
                forcefield_settings=forcefield_settings,
                integrator_settings=integrator_settings,
                thermo_settings=thermo_settings,
                cache=ffcache,
                has_solvent=solvent_comp is not None,
            )

            # c. force the creation of parameters
            # This is necessary because we need to have the FF templates
            # registered ahead of solvating the system.
            for smc, mol in chain(off_small_mols['stateA'],
                                  off_small_mols['stateB'],
                                  off_small_mols['both']):
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
        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
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
        # Block out oechem backend in system_generator calls to avoid
        # any issues with smiles roundtripping between rdkit and oechem
        with without_oechem_backend():
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

        # d. if a charge correction is necessary, select alchemical waters
        #    and transform them
        if alchem_settings.explicit_charge_correction:
            alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
                stateA_topology, stateA_positions,
                charge_difference,
                alchem_settings.explicit_charge_correction_cutoff,
            )
            _rfe_utils.topologyhelpers.handle_alchemical_waters(
                alchem_water_resids, stateB_topology, stateB_system,
                ligand_mappings, charge_difference,
                solvent_comp,
            )

        #  e. Finally get the positions
        stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings, stateA_topology, stateB_topology,
            old_positions=ensure_quantity(stateA_positions, 'openmm'),
            insert_positions=ensure_quantity(off_small_mols['stateB'][0][1].conformers[0], 'openmm'),
        )

        # 3. Create the hybrid topology
        # a. Get softcore potential settings
        if alchem_settings.softcore_LJ.lower() == 'gapsys':
            softcore_LJ_v2 = True
        elif alchem_settings.softcore_LJ.lower() == 'beutler':
            softcore_LJ_v2 = False
        # b. Get hybrid topology factory
        hybrid_factory = _rfe_utils.relative.HybridTopologyFactory(
            stateA_system, stateA_positions, stateA_topology,
            stateB_system, stateB_positions, stateB_topology,
            old_to_new_atom_map=ligand_mappings['old_to_new_atom_map'],
            old_to_new_core_atom_map=ligand_mappings['old_to_new_core_atom_map'],
            use_dispersion_correction=alchem_settings.use_dispersion_correction,
            softcore_alpha=alchem_settings.softcore_alpha,
            softcore_LJ_v2=softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchem_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchem_settings.turn_off_core_unique_exceptions,
        )

        # 4. Create lambda schedule
        # TODO - this should be exposed to users, maybe we should offer the
        # ability to print the schedule directly in settings?
        lambdas = _rfe_utils.lambdaprotocol.LambdaProtocol(
            functions=lambda_settings.lambda_functions,
            windows=lambda_settings.lambda_windows
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
                output_settings.output_indices
        )

        #  a. Create the multistate reporter
        # convert checkpoint_interval from time to iterations
        chk_intervals = settings_validation.convert_checkpoint_interval_to_iterations(
            checkpoint_interval=output_settings.checkpoint_interval,
            time_per_iteration=sampler_settings.time_per_iteration,
        )

        nc = shared_basepath / output_settings.output_filename
        chk = output_settings.checkpoint_storage_filename
        reporter = multistate.MultiStateReporter(
            storage=nc,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=chk_intervals,
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
                shared_basepath / output_settings.output_structure,
                bfactors=bfactors,
            )

        # 10. Get platform
        platform = _rfe_utils.compute.get_openmm_platform(
            protocol_settings.engine_settings.compute_platform
        )

        # 11. Set the integrator
        # a. Validate integrator settings for current system
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
            collision_rate=to_openmm(integrator_settings.langevin_collision_rate),
            n_steps=steps_per_iteration,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
        )

        # 12. Create sampler
        self.logger.info("Creating and setting up the sampler")
        rta_its, rta_min_its = settings_validation.convert_real_time_analysis_iterations(
            simulation_settings=sampler_settings,
        )
        # convert early_termination_target_error from kcal/mol to kT
        early_termination_target_error = settings_validation.convert_target_error_from_kcal_per_mole_to_kT(
            thermo_settings.temperature,
            sampler_settings.early_termination_target_error,
        )

        if sampler_settings.sampler_method.lower() == "repex":
            sampler = _rfe_utils.multistate.HybridRepexSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = _rfe_utils.multistate.HybridSAMSSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_minimum_iterations=rta_min_its,
                flatness_criteria=sampler_settings.sams_flatness_criteria,
                gamma0=sampler_settings.sams_gamma0,
            )
        elif sampler_settings.sampler_method.lower() == 'independent':
            sampler = _rfe_utils.multistate.HybridMultiStateSampler(
                mcmc_moves=integrator,
                hybrid_factory=hybrid_factory,
                online_analysis_interval=rta_its,
                online_analysis_target_error=early_termination_target_error,
                online_analysis_minimum_iterations=rta_min_its,
            )

        else:
            raise AttributeError(f"Unknown sampler {sampler_settings.sampler_method}")

        sampler.setup(
            n_replicas=sampler_settings.n_replicas,
            reporter=reporter,
            lambda_protocol=lambdas,
            temperature=to_openmm(thermo_settings.temperature),
            endstates=alchem_settings.endstate_dispersion_correction,
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

                sampler.minimize(max_iterations=sampler_settings.minimization_steps)

                # equilibrate
                if verbose:
                    self.logger.info("Running equilibration phase")

                sampler.equilibrate(
                    int(equil_steps / steps_per_iteration)  # type: ignore
                )

                # production
                if verbose:
                    self.logger.info("Running production phase")

                sampler.extend(
                    int(prod_steps / steps_per_iteration)  # type: ignore
                )

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
                fns = [shared_basepath / output_settings.output_filename,
                       shared_basepath / output_settings.checkpoint_storage_filename]
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

    @staticmethod
    def analyse(where) -> dict:
        # don't put energy analysis in here, it uses the open file reporter
        # whereas structural stuff requires that the file handle is closed
        analysis_out = where / 'structural_analysis.json'

        ret = subprocess.run(['openfe_analysis', 'RFE_analysis',
                              str(where), str(analysis_out)],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        if ret.returncode:
            return {'structural_analysis_error': ret.stderr}

        with open(analysis_out, 'rb') as f:
            data = json.load(f)

        savedir = pathlib.Path(where)
        if d := data['protein_2D_RMSD']:
            fig = plotting.plot_2D_rmsd(d)
            fig.savefig(savedir / "protein_2D_RMSD.png")
            plt.close(fig)
            f2 = plotting.plot_ligand_COM_drift(data['time(ps)'], data['ligand_wander'])
            f2.savefig(savedir / "ligand_COM_drift.png")
            plt.close(f2)

        f3 = plotting.plot_ligand_RMSD(data['time(ps)'], data['ligand_RMSD'])
        f3.savefig(savedir / "ligand_RMSD.png")
        plt.close(f3)

        return {'structural_analysis': data}

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])
        
        outputs = self.run(scratch_basepath=ctx.scratch,
                           shared_basepath=ctx.shared)

        analysis_outputs = self.analyse(ctx.shared)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs,
            **analysis_outputs,
        }
