# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""OpenMM Equilibrium AFE Protocol --- :mod:`openfe.protocols.openmm_afe.equil_afe_methods`
===========================================================================================

This module implements the necessary methodology toolking to run calculate an
absolute free energy transformation using OpenMM tools and one of the
following alchemical sampling methods:

* Hamiltonian Replica Exchange
* Self-adjusted mixture sampling
* Independent window sampling


.. versionadded:: 0.10.2


Running a Solvation Free Energy Calculation
-------------------------------------------

One use case of this Protocol is to carry out absolute solvation free energy
calculations. This involves capturing the free energy cost associated with
taking a small molecule from a solvent environment to gas phase.

In practice, because OpenMM currently only allows for charge annhilation when
using an exact treatment of charges using PME, this ends up requiring two
transformations. The first is carried out in solvent, where we annhilate the
charges of the ligand, and then decouple the LJ interactions. The second is
done in gas phase and involves recharging the ligand.

Here we provide a short overview on how such a thermodynamic cycle would be
achieved using this protocol.

Assuming we have a ligand of interest contained within an SDF file named
`ligands.sdf`, we can start by loading it into a SmallMoleculeComponent.


.. code-block::

    from gufe import SmallMoleculeComponent

    mol = SmallMoleculeComponent.from_sdf_file('ligand.sdf')


With this, we can next create ChemicalSystem objects for the four
different end states of our thermodynamic cycle.


.. code-block::

    from gufe import ChemicalSystem

    # State with a ligand in solvent
    ligand_solvent = ChemicalSystem({
        'ligand': mol, 'solvent': SolventComponent()
    })

    # State with only solvent
    solvent = ChemicalSystem({'solvent': SolventComponent()})

    # State with only the ligand in gas phase
    ligand_gas = ChemicalSystem({'ligand': mol})

    # State that is purely gas phase
    gas = ChemicalSystem({'ligand': mol})


Next we generate settings to run both solvent and gas phase transformations.
Aside form unique file names for the trajectory & checkpoint files, the main
difference in the settings is that we have to set the nonbonded method to be
`nocutoff` for gas phase and `pme` (the default) for periodic solvated systems.
Note: for everything else we use the default settings, howeve rin practice you
may find that much shorter simulation times may be adequate for gas phase
simulations.


.. code-block::

    solvent_settings = AbsoluteTransformProtocol._default_settings()
    solvent_settings.simulation_settings.output_filename = "ligand_solvent.nc"
    solvent_settings.simulation_settings.checkpoint_storage = "ligand_solvent_checkpoint.nc"

    gas_setttings = AbsoluteTransformProtocol._default_settings()
    gas_settings.simulation_settings.output_filename = "ligand_gas.nc"
    gas_settings.simulation_settings.checkpoint_storage = "ligand_gas_checkpoint.nc"

    # By default the nonbonded method is PME, this needs to be nocutoff for gas phase
    gas_settings.system_settings.nonbonded_method = 'nocutoff'


With this, we can create protocols and simulation DAGs for each leg of the
cycle. We pass to create the corresponding chemical end states of each leg
(e.g. ligand in solvent and pure solvent for the solvent transformation leg)
We note that no mapping is passed through to the Protocol. The Protocol
automatically compares the components present in the ChemicalSystems passed to
stateA and stateB and identifies any components missing either either of the
end states as undergoing an alchemical transformation.


.. code-block::

    solvent_transform = AbsoluteTransformProtocol(settings=solvent_settings)
    solvent_dag = solvent_transform.create(stateA=ligand_solvent, stateB=solvent, mapping=None)
    gas_transform = AbsoluteTransformProtocol(settings=gas_settings)
    gas_dag = solvent_transform.create(stateA=ligand_gas, stateB=gas, mapping=None)


Next we execute the transformations. By default, this will simulate 3 repeats
of both the ligand in solvent and ligand in gas transformations. Note: this
will take a while to run.


.. code-block::

    from gufe.protocols import execute_DAG
    solvent_data = execute_DAG(solvent_dag, shared='./solvent')
    gas_data = execute_DAG(gas_dag, shared='./gas')


Once completed, we gather the results and then get our estimate as the
difference between the gas and solvent transformations.


.. code-block::

    solvent_results = solvent_transform.gather([solvent_data,])
    gas_results = gas_transform.gather([gas_data,])
    dG = gas_results.get_estimate() - solvent_results.get_estimate()
    print(dG)


Current limitations
-------------------
* Disapearing molecules are only allowed in state A. Support for
  appearing molecules will be added in due course.
* Only one protein component is allowed per state. We ask that,
  users input all molecules intended to use an additive force field
  in the one ProteinComponent. This will likely change once OpenFF
  rosemary is released.
* Only small molecules are allowed to act as alchemical molecules.
  Alchemically changing protein or solvent components would induce
  perturbations which are too large to be handled by this Protocol.


Acknowledgements
----------------
* Originally based on the hydration.py in
  `espaloma <https://github.com/choderalab/espaloma_charge>`_


TODO
----
* Add in all the AlchemicalFactory and AlchemicalRegion kwargs
  as settings.
* Allow for a more flexible setting of Lambda regions.
* Add support for restraints.
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
from openfe.protocols.openmm_afe.equil_afe_settings import (
    AbsoluteTransformSettings, SystemSettings,
    SolvationSettings, AlchemicalSettings,
    AlchemicalSamplerSettings, OpenMMEngineSettings,
    IntegratorSettings, SimulationSettings,
)
from openfe.protocols.openmm_rfe._rfe_utils import compute
from ..openmm_utils import (
    system_validation, settings_validation, system_creation
)


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

    def get_rate_of_convergence(self):  # pragma: no-cover
        raise NotImplementedError


class AbsoluteSolvationProtocol(gufe.Protocol):
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
            solvent_system_settings=SystemSettings(),
            vacuum_system_settings=SystemSettings(nonbonded_method='nocutoff'),
            alchemical_settings=AlchemicalSettings(),
            alchemsampler_settings=AlchemicalSamplerSettings(),
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
        as starting from a ligand in solvent and ending up just in solvent.

        Parameters
        ----------
        stateA : ChemicalSystem
          The chemical system of end state A
        stateB : ChemicalSystem
          The chemical system of end state B

        Raises
        ------
        ValueError
          If stateB contains anything else but a SolventComponent.
          If stateA contains a ProteinComponent
        """
        if ((len(stateB) != 1) or
            (not isinstance(stateB.values()[0], SolventComponent))):
            errmsg = "Only a single SolventComponent is allowed in stateB"
            raise ValueError(errmsg)

        for comp in stateA.values():
            if isinstance(comp, ProteinComponent):
                errmsg = ("Protein components are not allow for "
                          "absolute solvation free energies")
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
        self._validate_solvation_endstates(stateA, stateB)
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
        assert vac_nonbonded_method.lower() != 'pme'

        # Get the name of the alchemical species
        alchname = alchem_comps['stateA'][0].name

        # Create list units for vacuum and solvent transforms

        solvent_units = [
            AbsoluteSolventTransformUnit(
                stateA=stateA, stateB=stateB,
                settings=self.settings,
                alchemical_components=alchemical_comps,
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
                alchemical_components=alchemical_comps,
                generation=0, repeat_id=i,
                name=(f"Absolute Solvation, {alchname} solvent leg: "
                      f"repeat {i} generation 0"),
            )
            for i in range(self.settings.alchemsampler_settings.n_repeats)
        ]

        return solvent_units + vacuum_units

    # TODO: update to match new unit list
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


class BaseAbsoluteTransformUnit(gufe.ProtocolUnit):
    """
    Base class for ligand absolute free energy transformations.
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

        # Currently the else is a dead branch, we will want to investigate the
        # possibility of using CutoffNonPeriodic at some point though (for RF)
        if nonbonded_method is not app.CutoffNonPeriodic:
            nonperiodic_kwargs = {
                'nonbondedMethod': app.NoCutoff,
            }
        else:  # pragma: no-cover
            nonperiodic_kwargs = periodic_kwargs

        system_generator = SystemGenerator(
            forcefields=protein_ffs,
            small_molecule_forcefield=small_ffs,
            forcefield_kwargs=forcefield_kwargs,
            nonperiodic_forcefield_kwargs=nonperiodic_kwargs,
            periodic_forcefield_kwargs=periodic_kwargs,
            cache=sim_settings.forcefield_cache,
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
                sim_settings.output_indices
        )

        # b. Create the multistate reporter
        reporter = multistate.MultiStateReporter(
            storage=basepath / sim_settings.output_filename,
            analysis_particle_indices=selection_indices,
            checkpoint_interval=sim_settings.checkpoint_interval.m,
            checkpoint_storage=basepath / sim_settings.checkpoint_storage,
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

        if not dry:  # pragma: no-cover
            # minimize
            if verbose:
                logger.info("minimizing systems")

            sampler.minimize(
                max_iterations=sim_settings.minimization_steps
            )

            # equilibrate
            if verbose:
                logger.info("equilibrating systems")

            sampler.equilibrate(int(equil_steps / mc_steps))  # type: ignore

            # production
            if verbose:
                logger.info("running production phase")

            sampler.extend(int(prod_steps / mc_steps))  # type: ignore

            # close reporter when you're done
            reporter.close()

            nc = basepath / sim_settings.output_filename
            chk = basepath / sim_settings.checkpoint_storage
            return {
                'nc': nc,
                'last_checkpoint': chk,
            }
        else:
            # close reporter when you're done, prevent file handle clashes
            reporter.close()

            # clean up the reporter file
            fns = [basepath / sim_settings.output_filename,
                   basepath / sim_settings.checkpoint_storage]
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
