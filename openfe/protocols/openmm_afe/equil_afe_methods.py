# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Equilibrium AFE Protocol using OpenMM + OpenMMTools

This module implements the necessary methodology toolking to run calculate an
absolute free energy transformation using OpenMM tools and one of the
following methods:
    - Hamiltonian Replica Exchange
    - Self-adjusted mixture sampling
    - Independent window sampling

Acknowledgements
----------------
* Originally based on a script from hydration.py in
  `espaloma <https://github.com/choderalab/espaloma_charge>`_

TODO
----
* Add support for restraints
* Improve this docstring by adding an example use case.

"""
from __future__ import annotations

import os
import logging

from collections import defaultdict
import gufe
from gufe.components import Component
import numpy as np
import numpy.typing as npt
import openmm
from openff.toolkit import Molecule as OFFMol
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm, ensure_quantity
from openmmtools import multistate
from openmmtools.states import (SamplerState,
                                create_thermodynamic_state_protocol,)
from openmmtools.alchemy import (AlchemicalRegion, AbsoluteAlchemicalFactory,
                                 AlchemicalState,)
from typing import Dict, List, Optional, Tuple
from openmm import app
from openmm import unit as omm_unit
from openmmforcefields.generators import SystemGenerator
import pathlib
from typing import Any, Iterable
import openmmtools
import uuid
import mdtraj as mdt

from gufe import (
    settings, ChemicalSystem, SmallMoleculeComponent,
    ProteinComponent, SolventComponent
)
from openfe.protocols.openmm_afe.afe_settings import (
    AbsoluteTransformSettings, SystemSettings,
    SolventSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings,
)
from openfe.protocols.openmm_rbfe._rbfe_utils import compute


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AbsoluteTransformProtocolResult(gufe.ProtocolResult):
    """Dict-like container for the output of a AbsoluteTransform"""
    def __init__(self, **data):
        super().__init__(**data)
        # TODO: Detect when we have extensions and stitch these together?
        if any(len(files['nc_paths']) > 2 for files in self.data['nc_files']):
            raise NotImplementedError("Can't stitch together results yet")

        self._analyzers = []
        for f in self.data['nc_files']:
            nc = f['nc_paths'][0]
            chk = f['checkpoint_paths'][0]
            reporter = multistate.MultiStateReporter(
                           storage=nc,
                           checkpoint_storage=chk)
            analyzer = multistate.MultiStateSamplerAnalyzer(reporter)
            self._analyzers.append(analyzer)

    def get_estimate(self):
        """Free energy difference of this transformation

        Returns
        -------
        dG : unit.Quantity
          The free energy difference between the first and last states. This is
          a Quantity defined with units.

        TODO
        ----
        * Check this holds up completely for SAMS.
        """
        dGs = []

        for analyzer in self._analyzers:
            # this returns:
            # (matrix of) estimated free energy difference
            # (matrix of) estimated statistical uncertainty (one S.D.)
            dG, _ = analyzer.get_free_energy()
            dG = (dG[0, -1] * analyzer.kT).in_units_of(
                omm_unit.kilocalories_per_mole)

            dGs.append(dG)

        avg_val = np.average([i.value_in_unit(dGs[0].unit) for i in dGs])

        return avg_val * dGs[0].unit

    def get_uncertainty(self):
        """The uncertainty/error in the dG value"""
        dGs = []

        for analyzer in self._analyzers:
            # this returns:
            # (matrix of) estimated free energy difference
            # (matrix of) estimated statistical uncertainty (one S.D.)
            dG, _ = analyzer.get_free_energy()
            dG = (dG[0, -1] * analyzer.kT).in_units_of(
                omm_unit.kilocalories_per_mole)

            dGs.append(dG)

        std_val = np.std([i.value_in_unit(dGs[0].unit) for i in dGs])

        return std_val * dGs[0].unit

    def get_rate_of_convergence(self):
        raise NotImplementedError


class AbsoluteTransformProtocol(gufe.Protocol):
    result_cls = AbsoluteTransformProtocolResult
    _settings: AbsoluteTransformSettings

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
        return AbsoluteTransformSettings(
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            system_settings=SystemSettings(),
            alchemical_settings=AlchemicalSettings(),
            alchemsampler_settings=AlchemicalSamplerSettings(),
            solvent_settings=SolventSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(
                timestep=4.0 * unit.femtosecond,
                n_steps=250 * unit.timestep,
            ),
            simulation_settings=SimulationSettings(
                equilibration_length=2.0 * unit.nanosecond,
                production_length=5.0 * unit.nanosecond,
            )
        )

    @staticmethod
    def _get_alchemical_components(
            stateA: ChemicalSystem,
            stateB: ChemicalSystem) -> Dict[str, List[Component]]:
        """
        Checks equality of ChemicalSystem components across both states and
        identify which components do not match.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A.
        stateB : ChemicalSystem
          The chemical system of end state B.

        Returns
        -------
        alchemical_components : Dictionary
            Dictionary containing a list of alchemical components for each
            state.
        """
        matched_components = {}
        alchemical_components: Dict[str, List[Any]] = {
            'stateA': [], 'stateB': []
        }

        for keyA, valA in stateA.components.items():
            for keyB, valB in stateB.components.items():
                if valA.to_dict() == valB.to_dict():
                    matched_components[keyA] = keyB
                    break

        # populate state A alchemical components
        for keyA in stateA.components.keys():
            if keyA not in matched_components.keys():
                alchemical_components['stateA'].append(keyA)

        # populate state B alchemical components
        for keyB in stateB.components.keys():
            if keyB not in matched_components.values():
                alchemical_components['stateB'].append(keyB)

        return alchemical_components

    @staticmethod
    def _validate_alchemical_components(
            stateA: ChemicalSystem,
            alchemical_components: Dict[str, List[str]]):
        """
        Checks that the ChemicalSystem alchemical components are correct.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A.
        alchemical_components : Dict[str, List[str]]
          Dictionary containing the alchemical components for
          stateA and stateB.

        Raises
        ------
        ValueError
          If there are alchemical components in state B.
          If there are non SmallMoleculeComponent alchemical species.

        Notes
        -----
        * Currently doesn't support alchemical components in state B.
        * Currently doesn't support alchemical components which are not
          SmallMoleculeComponents.
        """

        # Crash out if there are any alchemical components in state B for now
        if len(alchemical_components['stateB']) > 0:
            errmsg = ("Components appearing in state B are not "
                      "currently supported")
            raise ValueError(errmsg)

        # Crash out if any of the alchemical components are not
        # SmallMoleculeComponent
        for key in alchemical_components['stateA']:
            comp = stateA.components[key]
            if not isinstance(comp, SmallMoleculeComponent):
                errmsg = ("Non SmallMoleculeComponent alchemical species "
                          "are not currently supported")
                raise ValueError(errmsg)

    @staticmethod
    def _validate_solvent(state: ChemicalSystem, nonbonded_method: str):
        """
        Checks that the ChemicalSystem component has the right solvent
        composition with an input nonbonded_method.

        Parameters
        ----------
        state : ChemicalSystem
          The chemical system to inspect

        Raises
        ------
        ValueError
          If there are multiple SolventComponents in the ChemicalSystem
          or if there is a SolventComponent and
          `nonbonded_method` is `nocutoff`
        """
        solvents = 0
        for component in state.components.values():
            if isinstance(component, SolventComponent):
                if nonbonded_method.lower() == "nocutoff":
                    errmsg = (f"{nonbonded_method} cannot be used for vacuum "
                              "transformation")
                    raise ValueError(errmsg)
                solvents += 1

        if solvents > 1:
            errmsg = "Multiple SolventComponents found, only one is supported"
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

        # Checks on the inputs!
        # 1) check solvent compatibility
        nonbonded_method = self.settings.system_settings.nonbonded_method
        self._validate_solvent(stateA, nonbonded_method)
        self._validate_solvent(stateB, nonbonded_method)

        #  2) check your alchemical molecules
        #  Note: currently only SmallMoleculeComponents in state A are
        #  supported
        alchemical_comps = self._get_alchemical_components(stateA, stateB)
        self._validate_alchemical_components(stateA, alchemical_comps)

        # Get a list of names for all the alchemical molecules
        stateA_alchnames = ','.join(
            [stateA.components[c].name for c in alchemical_comps['stateA']]
        )

        # our DAG has no dependencies, so just list units
        units = [AbsoluteTransformUnit(
            stateA=stateA, stateB=stateB,
            settings=self.settings,
            alchemical_components=alchemical_comps,
            generation=0, repeat_id=i,
            name=f'Absolute {stateA_alchnames}: repeat {i} generation 0')
            for i in range(self.settings.alchemsampler_settings.n_repeats)]

        return units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> Dict[str, Any]:
        # result units will have a repeat_id and generation
        # first group according to repeat_id
        repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue
                rep = pu.outputs['repeat_id']
                gen = pu.outputs['generation']

                repeats[rep].append((
                    gen, pu.outputs['nc'],
                    pu.outputs['last_checkpoint']))

        data = []
        for rep_id, rep_data in sorted(repeats.items()):
            # then sort within a repeat according to generation
            nc_paths = [
                ncpath for gen, ncpath, nc_check in sorted(rep_data)
            ]
            chk_files = [
                nc_check for gen, ncpath, nc_check in sorted(rep_data)
            ]
            data.append({'nc_paths': nc_paths,
                         'checkpoint_paths': chk_files})

        return {
            'nc_files': data,
        }


class AbsoluteTransformUnit(gufe.ProtocolUnit):
    """
    Calculates an alchemical absolute free energy transformation of a ligand.
    """
    def __init__(self, *,
                 stateA: ChemicalSystem,
                 stateB: ChemicalSystem,
                 settings: settings.Settings,
                 alchemical_components: Dict[str, List[str]],
                 generation: int = 0,
                 repeat_id: int = 0,
                 name: Optional[str] = None,):
        """
        Parameters
        ----------
        stateA : ChemicalSystem
          ChemicalSystem containing the components defining the state at
          lambda 0.
        stateB : ChemicalSystem
          ChemicalSystem containing the components defining the state at
          lambda 1.
        settings : gufe.settings.Setings
          Settings for the Absolute Tranformation Protocol. This can be
          constructed by calling the
          :class:`AbsoluteTransformProtocol.get_default_settings` method
          to get a default set of settings.
        name : str, optional
          Human-readable identifier for this Unit
        repeat_id : int, optional
          Identifier for which repeat (aka replica/clone) this Unit is,
          default 0
        generation : int, optional
          Generation counter which keeps track of how many times this repeat
          has been extended, default 0.
        """
        super().__init__(
            name=name,
            stateA=stateA,
            stateB=stateB,
            settings=settings,
            alchemical_components=alchemical_components,
            repeat_id=repeat_id,
            generation=generation,
        )

    ParseCompRet = Tuple[
        Optional[SolventComponent], Optional[ProteinComponent],
        Dict[str, OFFMol],
    ]

    @staticmethod
    def _parse_components(state: ChemicalSystem) -> ParseCompRet:
        """
        Establish all necessary Components for the transformation.

        Parameters
        ----------
        state : ChemicalSystem
          Chemical system to get all necessary components from.

        Returns
        -------
        solvent_comp : Optional[SolventComponent]
          If it exists, the SolventComponent for the state, otherwise None.
        protein_comp : Optional[ProteinComponent]
          If it exists, the ProteinComponent for the state, otherwise None.
        openff_mols : Dict[str, openff.toolkit.Molecule]
          A dictionary of openff.toolkit Molecules for each
          SmallMoleculeComponent in the input state keyed by the original
          component name.

        Raises
        ------
        ValueError
          If there are more than one ProteinComponent

        TODO
        ----
        * Fix things so that we can have multiple ProteinComponents
        """
        # Is the system solvated?
        solvent_comp = None
        for comp in state.components.values():
            if isinstance(comp, SolventComponent):
                solvent_comp = comp

        # Is it complexed?
        # TODO: we intentionally crash if there's multiple proteins, fix this!
        protein_comp = None
        for comp in state.components.values():
            if isinstance(comp, ProteinComponent):
                if protein_comp is not None:
                    errmsg = "Multiple proteins are not currently supported"
                    raise ValueError(errmsg)
                protein_comp = comp

        # Get a dictionary of SmallMoleculeComponents as openff Molecules
        off_small_mols = {}
        for key, comp in state.components.items():
            if isinstance(comp, SmallMoleculeComponent):
                off_small_mols[key] = comp.to_openff()

        return solvent_comp, protein_comp, off_small_mols

    @staticmethod
    def _get_sim_steps(time: unit.Quantity, timestep: unit.Quantity,
                       mc_steps: int) -> unit.Quantity:
        """
        Get and validate the number of simulation steps

        Parameters
        ----------
        time : unit.Quantity
          Simulation time in femtoseconds.
        timestep : unit.Quantity
          Simulation timestep in femtoseconds.
        mc_steps : int
          Number of integration steps between MC moves.

        Returns
        -------
        steps : unit.Quantity
          Total number of integration steps

        Raises
        ------
        ValueError
          If the number of steps is not divisible by the number of mc_steps.
        """
        steps = round(time / timestep)

        if (steps.m % mc_steps) != 0:  # type: ignore
            errmsg = (f"Simulation time {time} should contain a number of "
                      "steps divisible by the number of integrator "
                      f"timesteps between MC moves {mc_steps}")
            ValueError(errmsg)

        return steps

    ModellerReturn = Tuple[app.Modeller, Dict[str, npt.NDArray]]

    @staticmethod
    def _get_omm_modeller(protein_comp: Optional[ProteinComponent],
                          solvent_comp: Optional[SolventComponent],
                          off_mols: Dict[str, OFFMol],
                          omm_forcefield: app.ForceField,
                          solvent_settings: settings.SettingsBaseModel,
                          ) -> ModellerReturn:
        """
        Generate an OpenMM Modeller class based on a potential input
        ProteinComponent, and a set of openff molecules.

        Parameters
        ----------
        protein_comp : Optional[ProteinComponent]
          Protein Component, if it exists.
        solvent_comp : Optional[ProteinCompoinent]
          Solvent COmponent, if it exists.
        off_mols : List[openff.toolkit.Molecule]
          List of small molecules as OpenFF Molecule.
        omm_forcefield : app.ForceField
          ForceField object for system.
        solvent_settings : settings.SettingsBaseModel
          Solventation settings

        Returns
        -------
        system_modeller : app.Modeller
          OpenMM Modeller object generated from ProteinComponent and
          OpenFF Molecules.
        component_resids : Dict[str, npt.NDArray]
          List of residue indices for each component in system.
        """
        component_resids = {}

        def _add_small_mol(compkey: str, mol: OFFMol,
                           system_modeller: app.Modeller,
                           comp_resids: Dict[str, npt.NDArray]):
            """
            Helper method to add off molecules to an existing Modeller
            object and update a dictionary tracking residue indices
            for each component.
            """
            omm_top = mol.to_topology().to_openmm()
            system_modeller.add(
                omm_top,
                ensure_quantity(mol.conformers[0], 'openmm')
            )

            nres = omm_top.getNumResidues()
            resids = [res.index for res in system_modeller.topology.residues()]
            comp_resids[key] = np.array(resids[-nres:])

        # If there's a protein in the system, we add it first to Modeller
        if protein_comp is not None:
            system_modeller = app.Modeller(protein_comp.to_openmm_topology(),
                                           protein_comp.to_openmm_positions())
            component_resids['protein'] = np.array(
                [res.index for res in system_modeller.topology.residues()]
            )

            for key, mol in off_mols.items():
                _add_small_mol(key, mol, system_modeller, component_resids)
        # Otherwise, we add the first small molecule, and then the rest
        else:
            mol_items = list(off_mols.items())

            system_modeller = app.Modeller(
                mol_items[0][1].to_topology().to_openmm(),
                ensure_quantity(mol_items[0][1].conformers[0], 'openmm')
            )
            component_resids[mol_items[0][0]] = np.array(
                [res.index for res in system_modeller.topology.residues()]
            )

            for key, mol in mol_items[1:]:
                _add_small_mol(key, mol, system_modeller, component_resids)

        # If there's solvent, add it and then set leftover resids to solvent
        if solvent_comp is not None:
            conc = solvent_comp.ion_concentration
            pos = solvent_comp.positive_ion
            neg = solvent_comp.negative_ion

            system_modeller.addSolvent(
                omm_forcefield,
                model=solvent_settings.solvent_model,
                padding=to_openmm(solvent_settings.solvent_padding),
                positiveIon=pos, negativeIon=neg,
                ionicStrength=to_openmm(conc),
            )

            all_resids = np.array(
                [res.index for res in system_modeller.topology.residues()]
            )

            existing_resids = np.concatenate(
                [resids for resids in component_resids.values()]
            )

            component_resids['solvent'] = np.setdiff1d(
                all_resids, existing_resids
            )

        return system_modeller, component_resids

    @staticmethod
    def _get_alchemical_indices(omm_top: openmm.Topology,
                                comp_resids: Dict[str, npt.NDArray],
                                alchem_comps: Dict[str, List[str]]
                                ) -> List[int]:
        """
        Get a list of atom indices for all the alchemical species

        Parameters
        ----------
        omm_top : openmm.Topology
          Topology of OpenMM System.
        comp_resids : Dict[str, npt.NDArray]
          A dictionary of residues for each component in the System.
        alchem_comps : Dict[str, List[str]]
          A dictionary of alchemical components for each end state.

        Return
        ------
        atom_ids : List[int]
          A list of atom indices for the alchemical species
        """

        # concatenate a list of residue indexes for all alchemical components
        residxs = np.concatenate(
            [comp_resids[key] for key in alchem_comps['stateA']]
        )

        # get the alchemicical residues from the topology
        alchres = [
            r for r in omm_top.residues() if r.index in residxs
        ]

        atom_ids = []

        for res in alchres:
            atom_ids.extend([at.index for at in res.atoms()])

        return atom_ids

    @staticmethod
    def _pre_minimize(system: openmm.System,
                      positions: omm_unit.Quantity) -> npt.NDArray:
        """
        Short CPU minization of System to avoid GPU NaNs

        Parameters
        ----------
        system : openmm.System
          An OpenMM System to minimize.
        positionns : openmm.unit.Quantity
          Initial positions for the system.

        Returns
        -------
        minimized_positions : npt.NDArray
          Minimized positions
        """
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(
                system, integrator,
                openmm.Platform.getPlatformByName('CPU'),
        )
        context.setPositions(positions)
        # Do a quick 100 steps minimization, usually avoids NaNs
        openmm.LocalEnergyMinimizer.minimize(
                context, maxIterations=100
        )
        state = context.getState(getPositions=True)
        minimized_positions = state.getPositions(asNumpy=True)
        return minimized_positions

    def run(self, dry=False, verbose=True, basepath=None) -> Dict[str, Any]:
        """Run the absolute free energy calculation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary alchemical
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.

        Attributes
        ----------
        solvent : Optional[SolventComponent]
          SolventComponent to be applied to the system
        protein : Optional[ProteinComponent]
          ProteinComponent for the system
        openff_mols : List[openff.Molecule]
          List of OpenFF Molecule objects for each SmallMoleculeComponent in
          the stateA ChemicalSystem
        """
        if verbose:
            logger.info("setting up alchemical system")

        # Get basepath
        if basepath is None:
            # use cwd
            basepath = pathlib.Path('.')

        # 0. General setup and settings dependency resolution step

        # a. Establish chemical system and their components
        stateA = self._inputs['stateA']
        alchem_comps = self._inputs['alchemical_components']
        # Get the relevant solvent & protein components & openff molecules
        solvent_comp, protein_comp, off_mols = self._parse_components(stateA)

        # b. Establish integration nsettings
        settings = self._inputs['settings']
        sim_settings = settings.simulation_settings
        timestep = settings.integrator_settings.timestep
        mc_steps = settings.integrator_settings.n_steps.m
        equil_time = sim_settings.equilibration_length.to('femtosecond')
        prod_time = sim_settings.production_length.to('femtosecond')
        equil_steps = self._get_sim_steps(equil_time, timestep, mc_steps)
        prod_steps = self._get_sim_steps(prod_time, timestep, mc_steps)

        # 1. Parameterise System
        # a. Set up SystemGenerator object
        ffsettings = settings.forcefield_settings
        protein_ffs = ffsettings.forcefields
        small_ffs = ffsettings.small_molecule_forcefield

        constraints = {
            'hbonds': app.HBonds,
            'none': None,
            'allbonds': app.AllBonds,
            'hangles': app.HAngles
            # vvv can be None so string it
        }[str(ffsettings.constraints).lower()]

        forcefield_kwargs = {
            'constraints': constraints,
            'rigidWater': ffsettings.rigid_water,
            'removeCMMotion': ffsettings.remove_com,
            'hydrogenMass': ffsettings.hydrogen_mass * omm_unit.amu,
        }

        nonbonded_method = {
            'pme': app.PME,
            'nocutoff': app.NoCutoff,
            'cutoffnonperiodic': app.CutoffNonPeriodic,
            'cutoffperiodic': app.CutoffPeriodic,
            'ewald': app.Ewald
        }[settings.system_settings.nonbonded_method.lower()]

        nonbonded_cutoff = to_openmm(
            settings.system_settings.nonbonded_cutoff
        )

        periodic_kwargs = {
            'nonbondedMethod': nonbonded_method,
            'nonbondedCutoff': nonbonded_cutoff,
        }

        if nonbonded_method is not app.CutoffNonPeriodic:
            nonperiodic_kwargs = {
                'nonbondedMethod': app.NoCutoff,
            }
        else:
            nonperiodic_kwargs = periodic_kwargs

        system_generator = SystemGenerator(
            forcefields=protein_ffs,
            small_molecule_forcefield=small_ffs,
            forcefield_kwargs=forcefield_kwargs,
            nonperiodic_forcefield_kwargs=nonperiodic_kwargs,
            periodic_forcefield_kwargs=periodic_kwargs,
            cache=settings.simulation_settings.forcefield_cache,
        )

        # Add a barostat if necessary note, was broken pre 0.11.2 of openmmff
        pressure = settings.thermo_settings.pressure
        temperature = settings.thermo_settings.temperature
        if solvent_comp is not None:
            barostat = openmm.MonteCarloBarostat(
                ensure_quantity(pressure, 'openmm'),
                ensure_quantity(temperature, 'openmm')
            )
            system_generator.barostat = barostat

        # force the creation of parameters for the small molecules
        # this is cached and shouldn't incur further cost
        for mol in off_mols.values():
            system_generator.create_system(mol.to_topology().to_openmm(),
                                           molecules=[mol])

        # b. Get OpenMM Modller + a dictionary of resids for each component
        system_modeller, comp_resids = self._get_omm_modeller(
            protein_comp, solvent_comp, off_mols, system_generator.forcefield,
            settings.solvent_settings
        )

        # c. Get OpenMM topology
        system_topology = system_modeller.getTopology()

        # d. Get initial positions (roundtrip via off_units to canocalize)
        positions = to_openmm(from_openmm(system_modeller.getPositions()))

        # d. Create System
        omm_system = system_generator.create_system(
            system_modeller.topology,
            molecules=list(off_mols.values())
        )

        # e. Get a list of indices for the alchemical species
        alchemical_indices = self._get_alchemical_indices(
            system_topology, comp_resids, alchem_comps
        )

        # 2. Pre-minimize System (Test + Avoid NaNs)
        positions = self._pre_minimize(omm_system, positions)

        # 3. Create the alchemical system
        # a. Get alchemical settings
        alchem_settings = settings.alchemical_settings

        # b. Set the alchemical region & alchemical factory
        # TODO: add support for all the variants here
        # TODO: check that adding indices this way works
        alchemical_region = AlchemicalRegion(
                alchemical_atoms=alchemical_indices,
        )
        alchemical_factory = AbsoluteAlchemicalFactory()
        alchemical_system = alchemical_factory.create_alchemical_system(
                omm_system, alchemical_region
        )

        # c. Create the lambda schedule
        # TODO: do this properly using LambdaProtocol
        # TODO: double check we definitely don't need to define
        #       temperature & pressure (pressure sure that's the case)
        lambdas = dict()
        n_elec = alchem_settings.lambda_elec_windows
        n_vdw = alchem_settings.lambda_vdw_windows + 1
        lambdas['lambda_electrostatics'] = np.concatenate(
                [np.linspace(1, 0, n_elec), np.linspace(0, 0, n_vdw)[1:]]
        )
        lambdas['lambda_sterics'] = np.concatenate(
                [np.linspace(1, 1, n_elec), np.linspace(1, 0, n_vdw)[1:]]
        )

        # d. Check that the lambda schedule matches n_replicas
        # TODO: undo for SAMS
        n_replicas = settings.alchemsampler_settings.n_replicas

        if n_replicas != (len(lambdas['lambda_sterics'])):
            errmsg = (f"Number of replicas {n_replicas} "
                      "does not equal the number of lambda windows ")
            raise ValueError(errmsg)

        # 4. Create compound states
        alchemical_state = AlchemicalState.from_system(alchemical_system)
        constants = dict()
        constants['temperature'] = ensure_quantity(temperature, 'openmm')
        if solvent_comp is not None:
            constants['pressure'] = ensure_quantity(pressure, 'openmm')
        cmp_states = create_thermodynamic_state_protocol(
                alchemical_system,
                protocol=lambdas,
                constants=constants,
                composable_states=[alchemical_state],
        )

        # 5. Create the sampler states
        # Fill up a list of sampler states all with the same starting state
        sampler_state = SamplerState(positions=positions)
        if omm_system.usesPeriodicBoundaryConditions():
            box = omm_system.getDefaultPeriodicBoxVectors()
            sampler_state.box_vectors = box

        sampler_states = [sampler_state for _ in cmp_states]

        # 6. Create the multistate reporter
        # a. Get the sub selection of the system to print coords for
        mdt_top = mdt.Topology.from_openmm(system_topology)
        selection_indices = mdt_top.select(
                settings.simulation_settings.output_indices
        )

        # b. Create the multistate reporter
        reporter = multistate.MultiStateReporter(
            storage=basepath / settings.simulation_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=settings.simulation_settings.checkpoint_interval.m,
            checkpoint_storage=basepath / settings.simulation_settings.checkpoint_storage,
        )

        # 7. Get platform and context caches
        platform = compute.get_openmm_platform(
            settings.engine_settings.compute_platform
        )

        # a. Create context caches (energy + sampler)
        #    Note: these needs to exist on the compute node
        energy_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        sampler_context_cache = openmmtools.cache.ContextCache(
            capacity=None, time_to_live=None, platform=platform,
        )

        # 8. Set the integrator
        # a. get integrator settings
        integrator_settings = settings.integrator_settings

        # b. create langevin integrator
        integrator = openmmtools.mcmc.LangevinSplittingDynamicsMove(
            timestep=to_openmm(integrator_settings.timestep),
            collision_rate=to_openmm(integrator_settings.collision_rate),
            n_steps=integrator_settings.n_steps.m,
            reassign_velocities=integrator_settings.reassign_velocities,
            n_restart_attempts=integrator_settings.n_restart_attempts,
            constraint_tolerance=integrator_settings.constraint_tolerance,
            splitting=integrator_settings.splitting
        )

        # 9. Create sampler
        sampler_settings = settings.alchemsampler_settings

        # Select the right sampler
        # Note: doesn't need else, settings already validates choices
        if sampler_settings.sampler_method.lower() == "repex":
            sampler = multistate.ReplicaExchangeSampler(
                mcmc_moves=integrator,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )
        elif sampler_settings.sampler_method.lower() == "sams":
            sampler = multistate.SAMSSampler(
                mcmc_moves=integrator,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations,
                flatness_criteria=sampler_settings.flatness_criteria,
                gamma0=sampler_settings.gamma0,
            )
        elif sampler_settings.sampler_method.lower() == 'independent':
            sampler = multistate.MultiStateSampler(
                mcmc_moves=integrator,
                online_analysis_interval=sampler_settings.online_analysis_interval,
                online_analysis_target_error=sampler_settings.online_analysis_target_error.m,
                online_analysis_minimum_iterations=sampler_settings.online_analysis_minimum_iterations
            )

        sampler.create(
                thermodynamic_states=cmp_states,
                sampler_states=sampler_states,
                storage=reporter
        )

        sampler.energy_context_cache = energy_context_cache
        sampler.sampler_context_cache = sampler_context_cache

        if not dry:
            # minimize
            if verbose:
                logger.info("minimizing systems")

            sampler.minimize(
                max_iterations=settings.simulation_settings.minimization_steps
            )

            # equilibrate
            if verbose:
                logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps.m / mc_steps))  # type: ignore

            # production
            if verbose:
                logger.info("running production phase")

            sampler.extend(int(prod_steps.m / mc_steps))  # type: ignore

            # close reporter when you're done
            reporter.close()

            nc = basepath / settings.simulation_settings.output_filename
            chk = basepath / settings.simulation_settings.checkpoint_storage
            return {
                'nc': nc,
                'last_checkpoint': chk,
            }
        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [basepath / settings.simulation_settings.output_filename,
                   basepath / settings.simulation_settings.checkpoint_storage]
            for fn in fns:
                os.remove(fn)
            return {'debug': {'sampler': sampler}}

    def _execute(
        self, ctx: gufe.Context, **kwargs,
    ) -> Dict[str, Any]:
        # create directory for *this* unit within the context of the *DAG*
        # stops output files mashing into each other within a DAG
        myid = uuid.uuid4()
        mypath = pathlib.Path(os.path.join(ctx.shared, str(myid)))
        mypath.mkdir(parents=True, exist_ok=False)

        outputs = self.run(basepath=mypath)

        return {
            'repeat_id': self._inputs['repeat_id'],
            'generation': self._inputs['generation'],
            **outputs
        }
