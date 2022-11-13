# This code is a slightly modified version of the HybridTopologyFactory code
# from https://github.com/choderalab/perses
# The eventual goal is to move a version of this towards openmmtools
# LICENSE: MIT

import logging
import openmm
from openmm import unit
import numpy as np
import copy
import itertools
# OpenMM constant for Coulomb interactions (implicitly in md_unit_system units)
from openmmtools.constants import ONE_4PI_EPS0
import mdtraj as mdt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HybridTopologyFactory:
    """
    This class generates a hybrid topology based on two input systems and an
    atom mapping. For convenience the states are called "old" and "new"
    respectively, defining the starting and end states along the alchemical
    transformation.

    The input systems are assumed to have:
        1. The total number of molecules
        2. The same coordinates for equivalent atoms

    Atoms in the resulting hybrid system are treated as being from one
    of four possible types:

    unique_old_atom : These atoms are not mapped and only present in the old
        system. Their interactions will be on for lambda=0, off for lambda=1
    unique_new_atom : These atoms are not mapped and only present in the new
        system. Their interactions will be off for lambda=0, on for lambda=1
    core_atom : These atoms are mapped between the two end states, and are
        part of a residue that is changing alchemically. Their interactions
        will be those corresponding to the old system at lambda=0, and those
        corresponding to the new system at lambda=1
    environment_atom : These atoms are mapped between the two end states, and
        are not part of a residue undergoing an alchemical change. Their
        interactions are always on and are alchemically unmodified.

    Properties
    ----------
    hybrid_system : openmm.System
        The hybrid system for simulation
    new_to_hybrid_atom_map : dict of int : int
        The mapping of new system atoms to hybrid atoms
    old_to_hybrid_atom_map : dict of int : int
        The mapping of old system atoms to hybrid atoms
    hybrid_positions : [n, 3] np.ndarray
        The positions of the hybrid system
    hybrid_topology : mdtraj.Topology
        The topology of the hybrid system
    omm_hybrid_topology : openmm.app.Topology
        The OpenMM topology object corresponding to the hybrid system

    .. warning :: This API is experimental and subject to change.

    Notes
    -----
    * Logging has been removed and will be revamped at a later date.
    * The ability to define custom functions has been removed for now.
    * Neglected angle terms have been removed for now.
    * RMSD restraint option has been removed for now.
    * Endstate support has been removed for now.
    * Bond softening has been removed for now.
    * Unused InteractionGroup code paths have been removed.

    TODO
    ----
    * Document how positions for hybrid system are constructed.
    * Allow support for annealing in omitted terms.
    * Implement omitted terms (this was not available in the original class).

    """

    def __init__(self,
                 old_system, old_positions, old_topology,
                 new_system, new_positions, new_topology,
                 old_to_new_atom_map, old_to_new_core_atom_map,
                 use_dispersion_correction=False,
                 softcore_alpha=0.5,
                 softcore_LJ_v2=True,
                 softcore_LJ_v2_alpha=0.85,
                 softcore_electrostatics=True,
                 softcore_electrostatics_alpha=0.3,
                 softcore_sigma_Q=1.0,
                 interpolate_old_and_new_14s=False,
                 flatten_torsions=False,
                 impose_virtual_bonds=True,
                 **kwargs):
        """
        Initialize the Hybrid topology factory.

        Parameters
        ----------
        old_system : openmm.System
            OpenMM system defining the "old" (i.e. starting) state.
        old_positions : [n,3] np.ndarray of float
            The positions of the "old system".
        old_topology : openmm.Topology
            OpenMM topology defining the "old" state.
        new_system: opemm.System
            OpenMM system defining the "new" (i.e. end) state.
        new_positions : [m,3] np.ndarray of float
            The positions of the "new system"
        new_topology : openmm.Topology
            OpenMM topology defining the "new" state.
        old_to_new_atom_map : dict of int : int
            Dictionary of corresponding atoms between the old and new systems.
            Unique atoms are not included in this atom map.
        old_to_new_core_atom_map : dict of int : int
            Dictionary of corresponding atoms between the alchemical "core
            atoms" (i.e. residues which are changing) between the old and
            new systems.
        use_dispersion_correction : bool, default False
            Whether to use the long range correction in the custom sterics
            force. This can be very expensive for NCMC.
        softcore_alpha: float, default None
            "alpha" parameter of softcore sterics, default 0.5.
        softcore_LJ_v2 : bool, default True
            Implement the softcore LJ as defined by Gapsys et al. JCTC 2012.
        softcore_LJ_v2_alpha : float, default 0.85
            Softcore alpha parameter for LJ v2
        softcore_electrostatics : bool, default True
            Use softcore electrostatics as defined by Gapsys et al. JCTC 2021.
        softcore_electrostatics_alpha : float, default 0.3
            Softcore alpha parameter for softcore electrostatics.
        softcore_sigma_Q : float, default 1.0
            Softcore sigma parameter for softcore electrostatics.
        interpolate_old_and_new_14s : bool, default False
            Whether to turn off interactions for new exceptions (not just
            1,4s) at lambda = 0 and old exceptions at lambda = 1; if False,
            they are present in the nonbonded force.
        flatten_torsions : bool, default False
            If True, torsion terms involving `unique_new_atoms` will be
            scaled such that at lambda=0,1, the torsion term is turned off/on
            respectively. The opposite is true for `unique_old_atoms`.
        impose_virtual_bonds : bool, default True
            If True, adds a virtual bond between non-contiguous protein and
            ligand components to ensure that they are imaged together.
        """

        # Assign system positions and force
        # IA - Are deep copies really needed here?
        self._old_system = copy.deepcopy(old_system)
        self._old_positions = old_positions
        self._old_topology = old_topology
        self._new_system = copy.deepcopy(new_system)
        self._new_positions = new_positions
        self._new_topology = new_topology
        self._hybrid_system_forces = dict()

        # Set mappings (full, core, and env maps)
        self._set_mappings(old_to_new_atom_map, old_to_new_core_atom_map)

        # Other options
        self._use_dispersion_correction = use_dispersion_correction
        self._interpolate_14s = interpolate_old_and_new_14s
        self._flatten_torsions = flatten_torsions

        # Sofcore options
        self._softcore_alpha = softcore_alpha
        self._check_bounds(softcore_alpha, "softcore_alpha")  # [0,1] check

        self._softcore_LJ_v2 = softcore_LJ_v2
        if self._softcore_LJ_v2:
            self._check_bounds(softcore_LJ_v2_alpha, "softcore_LJ_v2_alpha")
            self._softcore_LJ_v2_alpha = softcore_LJ_v2_alpha

        self._softcore_electrostatics = softcore_electrostatics
        if self._softcore_electrostatics:
            self._softcore_electrostatics_alpha = softcore_electrostatics_alpha
            self._check_bounds(softcore_electrostatics_alpha,
                               "softcore_electrostatics_alpha")
            self._softcore_sigma_Q = softcore_sigma_Q
            self._check_bounds(softcore_sigma_Q, "softcore_sigma_Q")

        # TODO: end __init__ here and move everything else to
        # create_hybrid_system() or equivalent

        self._check_and_store_system_forces()

        logger.info("creating hybrid system")
        # Create empty system that will become the hybrid system
        self._hybrid_system = openmm.System()

        # Add particles to system
        self._add_particles()

        # Add box + barostat
        self._handle_box()

        # Assign atoms to one of the classes described in the class docstring
        # Renamed from original _determine_atom_classes
        self._set_atom_classes()

        # Construct dictionary of exceptions in old and new systems
        self._old_system_exceptions = self._generate_dict_from_exceptions(
            self._old_system_forces['NonbondedForce'])
        self._new_system_exceptions = self._generate_dict_from_exceptions(
            self._new_system_forces['NonbondedForce'])

        # check for exceptions clashes between unique and env atoms
        self._validate_disjoint_sets()

        logger.info("setting force field terms")
        # Copy constraints, checking to make sure they are not changing
        self._handle_constraints()

        # Copy over relevant virtual sites - pick up refactor from here
        self._handle_virtual_sites()

        # TODO - move to a single method call? Would be good to group these
        # Call each of the force methods to add the corresponding force terms
        # and prepare the forces:
        self._add_bond_force_terms()

        self._add_angle_force_terms()

        self._add_torsion_force_terms()

        has_nonbonded_force = ('NonbondedForce' in self._old_system_forces or
                               'NonbondedForce' in self._new_system_forces)

        if has_nonbonded_force:
            self._add_nonbonded_force_terms()

        # Call each force preparation method to generate the actual
        # interactions that we need:
        logger.info("adding forces")
        self._handle_harmonic_bonds()

        self._handle_harmonic_angles()

        self._handle_periodic_torsion_force()

        if has_nonbonded_force:
            self._handle_nonbonded()
            if not (len(self._old_system_exceptions.keys()) == 0 and
                    len(self._new_system_exceptions.keys()) == 0):
                self._handle_old_new_exceptions()

        # Get positions for the hybrid
        self._hybrid_positions = self._compute_hybrid_positions()

        # Get an MDTraj topology for writing
        self._hybrid_topology = self._create_mdtraj_topology()
        logger.info("DONE")

        # Add virtual bonds to ensure system is imaged together
        if impose_virtual_bonds:
            self._impose_virtual_bonds()

    @staticmethod
    def _check_bounds(value, varname, minmax=(0, 1)):
        """
        Convenience method to check the bounds of a value.

        Parameters
        ----------
        value : float
            Value to evaluate.
        varname : str
            Name of value to raise in error message
        minmax : tuple
            Two element tuple with the lower and upper bounds to check.

        Raises
        ------
        AssertionError
            If value is lower or greater than bounds.
        """
        if value < minmax[0] or value > minmax[1]:
            raise AssertionError(f"{varname} is not in {minmax}")

    @staticmethod
    def _invert_dict(dictionary):
        """
        Convenience method to invert a dictionary (since we do it so often).

        Paramters:
        ----------
        dictionary : dict
            Dictionary you want to invert
        """
        return {v: k for k, v in dictionary.items()}

    def _set_mappings(self, old_to_new_map, core_old_to_new_map):
        """
        Parameters
        ----------
        old_to_new_map : dict of int : int
            Dictionary mapping atoms between the old and new systems.

        Notes
        -----
        * For now this directly sets the system, core and env old_to_new_map,
          new_to_old_map, an empty new_to_hybrid_map and an empty
          old_to_hybrid_map. In the future this will be moved to the one
          dictionary to make things a lot less confusing.
        """
        self._old_to_new_map = old_to_new_map
        self._core_old_to_new_map = core_old_to_new_map
        self._new_to_old_map = self._invert_dict(old_to_new_map)
        self._core_new_to_old_map = self._invert_dict(core_old_to_new_map)
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}

        # Get unique atoms
        # old system first
        self._unique_old_atoms = []
        for particle_idx in range(self._old_system.getNumParticles()):
            if particle_idx not in self._old_to_new_map.keys():
                self._unique_old_atoms.append(particle_idx)

        self._unique_new_atoms = []
        for particle_idx in range(self._new_system.getNumParticles()):
            if particle_idx not in self._new_to_old_map.keys():
                self._unique_new_atoms.append(particle_idx)

        # Get env atoms (i.e. atoms mapped not in core)
        self._env_old_to_new_map = {}
        for key, value in old_to_new_map.items():
            if key not in self._core_old_to_new_map.keys():
                self._env_old_to_new_map[key] = value

        self._env_new_to_old_map = self._invert_dict(self._env_old_to_new_map)

        # IA - Internal check for now (move to test later)
        num_env = len(self._env_old_to_new_map.keys())
        num_core = len(self._core_old_to_new_map.keys())
        num_total = len(self._old_to_new_map.keys())
        assert num_env + num_core == num_total

    def _check_and_store_system_forces(self):
        """
        Conveniently stores the system forces and checks that no unknown
        forces exist.
        """

        def _check_unknown_forces(forces, system_name):
            # TODO: double check that CMMotionRemover is ok being here
            known_forces = {'HarmonicBondForce', 'HarmonicAngleForce',
                            'PeriodicTorsionForce', 'NonbondedForce',
                            'MonteCarloBarostat', 'CMMotionRemover'}

            force_names = forces.keys()
            unknown_forces = set(force_names) - set(known_forces)
            if unknown_forces:
                errmsg = (f"Unknown forces {unknown_forces} encountered in "
                          f"{system_name} system")
                raise ValueError(errmsg)

        # Prepare dicts of forces, which will be useful later
        # TODO: Store this as self._system_forces[name], name in ('old',
        # 'new', 'hybrid') for compactness
        self._old_system_forces = {type(force).__name__: force for force in
                                   self._old_system.getForces()}
        _check_unknown_forces(self._old_system_forces, 'old')
        self._new_system_forces = {type(force).__name__: force for force in
                                   self._new_system.getForces()}
        _check_unknown_forces(self._new_system_forces, 'new')

        # TODO: check if this is actually used much, otherwise ditch it
        # Get and store the nonbonded method from the system:
        self._nonbonded_method = self._old_system_forces['NonbondedForce'].getNonbondedMethod()

    def _add_particles(self):
        """
        Adds particles to the hybrid system.

        This does not copy over interactions, but does copy over the masses.

        Note
        ----
        * If there is a difference in masses between the old and new systems
          the average mass of the two is used.

        TODO
        ----
        * Verify if we should just not allow elemental changes, current
          behaviour reflects original perses code.
        """
        # Begin by copying all particles in the old system
        for particle_idx in range(self._old_system.getNumParticles()):
            mass_old = self._old_system.getParticleMass(particle_idx)

            if particle_idx in self._old_to_new_map.keys():
                particle_idx_new_system = self._old_to_new_map[particle_idx]
                mass_new = self._new_system.getParticleMass(
                    particle_idx_new_system)
                # Take the average of the masses if the atom is mapped
                particle_mass = (mass_old + mass_new) / 2
            else:
                particle_mass = mass_old

            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._old_to_hybrid_map[particle_idx] = hybrid_idx

            # If the particle index in question is mapped, make sure to add it
            # to the new to hybrid map as well.
            if particle_idx in self._old_to_new_map.keys():
                self._new_to_hybrid_map[particle_idx_new_system] = hybrid_idx

        # Next, add the remaining unique atoms from the new system to the
        # hybrid system and map accordingly.
        for particle_idx in self._unique_new_atoms:
            particle_mass = self._new_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._new_to_hybrid_map[particle_idx] = hybrid_idx

        # Create the opposite atom maps for later use (nonbonded processing)
        self._hybrid_to_old_map = self._invert_dict(self._old_to_hybrid_map)
        self._hybrid_to_new_map = self._invert_dict(self._new_to_hybrid_map)

    def _handle_box(self):
        """
        Copies over the barostat and box vectors as necessary.
        """
        # Check that if there is a barostat in the old system,
        # it is added to the hybrid system
        if "MonteCarloBarostat" in self._old_system_forces.keys():
            barostat = copy.deepcopy(
                self._old_system_forces["MonteCarloBarostat"])
            self._hybrid_system.addForce(barostat)

        # Copy over the box vectors from the old system
        box_vectors = self._old_system.getDefaultPeriodicBoxVectors()
        self._hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)

    def _set_atom_classes(self):
        """
        This method determines whether each atom belongs to unique old,
        unique new, core, or environment, as defined in the class docstring.
        All indices are indices in the hybrid system.
        """
        self._atom_classes = {'unique_old_atoms': set(),
                              'unique_new_atoms': set(),
                              'core_atoms': set(),
                              'environment_atoms': set()}

        # First, find the unique old atoms
        for atom_idx in self._unique_old_atoms:
            hybrid_idx = self._old_to_hybrid_map[atom_idx]
            self._atom_classes['unique_old_atoms'].add(hybrid_idx)

        # Then the unique new atoms
        for atom_idx in self._unique_new_atoms:
            hybrid_idx = self._new_to_hybrid_map[atom_idx]
            self._atom_classes['unique_new_atoms'].add(hybrid_idx)

        # The core atoms:
        core_atoms = []
        for new_idx, old_idx in self._core_new_to_old_map.items():
            new_to_hybrid_idx = self._new_to_hybrid_map[new_idx]
            old_to_hybrid_idx = self._old_to_hybrid_map[old_idx]
            if new_to_hybrid_idx != old_to_hybrid_idx:
                errmsg = (f"there is an index collision in hybrid indices of "
                          f"the core atom map: {self._core_new_to_old_map}")
                raise AssertionError(errmsg)
            core_atoms.append(new_to_hybrid_idx)

        # The environment atoms:
        env_atoms = []
        for new_idx, old_idx in self._env_new_to_old_map.items():
            new_to_hybrid_idx = self._new_to_hybrid_map[new_idx]
            old_to_hybrid_idx = self._old_to_hybrid_map[old_idx]
            if new_to_hybrid_idx != old_to_hybrid_idx:
                errmsg = (f"there is an index collion in hybrid indices of "
                          f"the environment atom map: "
                          f"{self._env_new_to_old_map}")
                raise AssertionError(errmsg)
            env_atoms.append(new_to_hybrid_idx)

        # TODO - this is weirdly done and double assignments - fix
        self._atom_classes['core_atoms'] = set(core_atoms)
        self._atom_classes['environment_atoms'] = set(env_atoms)

    @staticmethod
    def _generate_dict_from_exceptions(force):
        """
        This is a utility function to generate a dictionary of the form
        (particle1_idx, particle2_idx) : [exception parameters].
        This will facilitate access and search of exceptions.

        Parameters
        ----------
        force : openmm.NonbondedForce object
            a force containing exceptions

        Returns
        -------
        exceptions_dict : dict
            Dictionary of exceptions
        """
        exceptions_dict = {}

        for exception_index in range(force.getNumExceptions()):
            [index1, index2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
            exceptions_dict[(index1, index2)] = [chargeProd, sigma, epsilon]

        return exceptions_dict

    def _validate_disjoint_sets(self):
        """
        Conduct a sanity check to make sure that the hybrid maps of the old
        and new system exception dict keys do not contain both environment
        and unique_old/new atoms.

        TODO: repeated code - condense
        """
        for old_indices in self._old_system_exceptions.keys():
            hybrid_indices = (self._old_to_hybrid_map[old_indices[0]],
                              self._old_to_hybrid_map[old_indices[1]])
            old_env_intersection = set(old_indices).intersection(
                self._atom_classes['environment_atoms'])
            if old_env_intersection:
                if set(old_indices).intersection(
                    self._atom_classes['unique_old_atoms']
                ):
                    errmsg = (f"old index exceptions {old_indices} include "
                              "unique old and environment atoms, which is "
                              "disallowed")
                    raise AssertionError(errmsg)

        for new_indices in self._new_system_exceptions.keys():
            hybrid_indices = (self._new_to_hybrid_map[new_indices[0]],
                              self._new_to_hybrid_map[new_indices[1]])
            new_env_intersection = set(hybrid_indices).intersection(
                self._atom_classes['environment_atoms'])
            if new_env_intersection:
                if set(hybrid_indices).intersection(
                    self._atom_classes['unique_new_atoms']
                ):
                    errmsg = (f"new index exceptions {new_indices} include "
                              "unique new and environment atoms, which is "
                              "dissallowed")
                    raise AssertionError

    def _handle_constraints(self):
        """
        This method adds relevant constraints from the old and new systems.

        First, all constraints from the old systenm are added.
        Then, constraints to atoms unique to the new system are added.

        TODO: condense duplicated code
        """
        # lengths of constraints already added
        constraint_lengths = dict()

        # old system
        hybrid_map = self._old_to_hybrid_map
        for const_idx in range(self._old_system.getNumConstraints()):
            at1, at2, length = self._old_system.getConstraintParameters(
                const_idx)
            hybrid_atoms = tuple(sorted([hybrid_map[at1], hybrid_map[at2]]))
            if hybrid_atoms not in constraint_lengths.keys():
                self._hybrid_system.addConstraint(hybrid_atoms[0],
                                                  hybrid_atoms[1], length)
                constraint_lengths[hybrid_atoms] = length
            else:

                if constraint_lengths[hybrid_atoms] != length:
                    raise AssertionError('constraint length is changing')

        # new system
        hybrid_map = self._new_to_hybrid_map
        for const_idx in range(self._new_system.getNumConstraints()):
            at1, at2, length = self._new_system.getConstraintParameters(
                const_idx)
            hybrid_atoms = tuple(sorted([hybrid_map[at1], hybrid_map[at2]]))
            if hybrid_atoms not in constraint_lengths.keys():
                self._hybrid_system.addConstraint(hybrid_atoms[0],
                                                  hybrid_atoms[1], length)
                constraint_lengths[hybrid_atoms] = length
            else:
                if constraint_lengths[hybrid_atoms] != length:
                    raise AssertionError('constraint length is changing')

    def _handle_virtual_sites(self):
        """
        Ensure that all virtual sites in old and new system are copied over to
        the hybrid system. Note that we do not support virtual sites in the
        changing region.

        TODO - remerge into a single loop
        TODO - check that it's fine to double count here (even so, there's
               an optimisation that could be done here...)
        """
        # old system
        # Loop through virtual sites
        for particle_idx in range(self._old_system.getNumParticles()):
            if self._old_system.isVirtualSite(particle_idx):
                # If it's a virtual site, make sure it is not in the unique or
                # core atoms, since this is currently unsupported
                hybrid_idx = self._old_to_hybrid_map[particle_idx]
                if hybrid_idx not in self._atom_classes['environment_atoms']:
                    errmsg = ("Virtual sites in changing residue are "
                              "unsupported.")
                    raise ValueError(errmsg)
                else:
                    virtual_site = self._old_system.getVirtualSite(
                        particle_idx)
                    self._hybrid_system.setVirtualSite(hybrid_idx,
                                                       virtual_site)

        # new system
        # Loop through virtual sites
        for particle_idx in range(self._new_system.getNumParticles()):
            if self._new_system.isVirtualSite(particle_idx):
                # If it's a virtual site, make sure it is not in the unique or
                # core atoms, since this is currently unsupported
                hybrid_idx = self._new_to_hybrid_map[particle_idx]
                if hybrid_idx not in self._atom_classes['environment_atoms']:
                    errmsg = ("Virtual sites in changing residue are "
                              "unsupported.")
                    raise ValueError(errmsg)
                else:
                    virtual_site = self._new_system.getVirtualSite(
                        particle_idx)
                    self._hybrid_system.setVirtualSite(hybrid_idx,
                                                       virtual_site)

    def _add_bond_force_terms(self):
        """
        This function adds the appropriate bond forces to the system
        (according to groups defined in the main class docstring). Note that
        it does _not_ add the particles to the force. It only adds the force
        to facilitate another method adding the particles to the force.

        Notes
        -----
        * User defined functions have been removed for now.
        """
        core_energy_expression = '(K/2)*(r-length)^2;'
        # linearly interpolate spring constant
        core_energy_expression += 'K = (1-lambda_bonds)*K1 + lambda_bonds*K2;'
        # linearly interpolate bond length
        core_energy_expression += 'length = (1-lambda_bonds)*length1 + lambda_bonds*length2;'

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('length1')  # old bond length
        custom_core_force.addPerBondParameter('K1')  # old spring constant
        custom_core_force.addPerBondParameter('length2')  # new bond length
        custom_core_force.addPerBondParameter('K2')  # new spring constant

        custom_core_force.addGlobalParameter('lambda_bonds', 0.0)

        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_bond_force'] = custom_core_force

        # Add a bond force for environment and unique atoms (bonds are never
        # scaled for these):
        standard_bond_force = openmm.HarmonicBondForce()
        self._hybrid_system.addForce(standard_bond_force)
        self._hybrid_system_forces['standard_bond_force'] = standard_bond_force

    def _add_angle_force_terms(self):
        """
        This function adds the appropriate angle force terms to the hybrid
        system. It does not add particles or parameters to the force; this is
        done elsewhere.

        Notes
        -----
        * User defined functions have been removed for now.
        * Neglected angle terms have been removed for now.
        """
        energy_expression = '(K/2)*(theta-theta0)^2;'
        # linearly interpolate spring constant
        energy_expression += 'K = (1.0-lambda_angles)*K_1 + lambda_angles*K_2;'
        # linearly interpolate equilibrium angle
        energy_expression += 'theta0 = (1.0-lambda_angles)*theta0_1 + lambda_angles*theta0_2;'

        # Create the force and add relevant parameters
        custom_core_force = openmm.CustomAngleForce(energy_expression)
        # molecule1 equilibrium angle
        custom_core_force.addPerAngleParameter('theta0_1')
        # molecule1 spring constant
        custom_core_force.addPerAngleParameter('K_1')
        # molecule2 equilibrium angle
        custom_core_force.addPerAngleParameter('theta0_2')
        # molecule2 spring constant
        custom_core_force.addPerAngleParameter('K_2')

        custom_core_force.addGlobalParameter('lambda_angles', 0.0)

        # Add the force to the system and the force dict.
        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_angle_force'] = custom_core_force

        # Add an angle term for environment/unique interactions -- these are
        # never scaled
        standard_angle_force = openmm.HarmonicAngleForce()
        self._hybrid_system.addForce(standard_angle_force)
        self._hybrid_system_forces['standard_angle_force'] = standard_angle_force

    def _add_torsion_force_terms(self):
        """
        This function adds the appropriate PeriodicTorsionForce terms to the
        system. Core torsions are interpolated, while environment and unique
        torsions are always on.

        Notes
        -----
        * User defined functions have been removed for now.
        * Options for add_custom_core_force (default True) and
          add_unique_atom_torsion_force (default True) have been removed for
          now.
        """
        energy_expression = '(1-lambda_torsions)*U1 + lambda_torsions*U2;'
        energy_expression += 'U1 = K1*(1+cos(periodicity1*theta-phase1));'
        energy_expression += 'U2 = K2*(1+cos(periodicity2*theta-phase2));'

        # Create the force and add the relevant parameters
        custom_core_force = openmm.CustomTorsionForce(energy_expression)
        # molecule1 periodicity
        custom_core_force.addPerTorsionParameter('periodicity1')
        # molecule1 phase
        custom_core_force.addPerTorsionParameter('phase1')
        # molecule1 spring constant
        custom_core_force.addPerTorsionParameter('K1')
        # molecule2 periodicity
        custom_core_force.addPerTorsionParameter('periodicity2')
        # molecule2 phase
        custom_core_force.addPerTorsionParameter('phase2')
        # molecule2 spring constant
        custom_core_force.addPerTorsionParameter('K2')

        custom_core_force.addGlobalParameter('lambda_torsions', 0.0)

        # Add the force to the system
        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['custom_torsion_force'] = custom_core_force

        # Create and add the torsion term for unique/environment atoms
        unique_atom_torsion_force = openmm.PeriodicTorsionForce()
        self._hybrid_system.addForce(unique_atom_torsion_force)
        self._hybrid_system_forces['unique_atom_torsion_force'] = unique_atom_torsion_force

    @staticmethod
    def _nonbonded_custom(v2):
        """
        Get a part of the nonbonded energy expression when there is no cutoff.

        Parameters
        ----------
        v2 : bool
            Whether to use the softcore methods as defined by Gapsys et al.
            JCTC 2012.

        Returns
        -------
        sterics_energy_expression : str
            The energy expression for U_sterics
        electrostatics_energy_expression : str
            The energy expression for electrostatics

        TODO
        ----
        * Move to a dictionary or equivalent.
        """
        # Soft-core Lennard-Jones
        if v2:
            sterics_energy_expression = "U_sterics = select(step(r - r_LJ), 4*epsilon*x*(x-1.0), U_sterics_quad);"
            sterics_energy_expression += "U_sterics_quad = Force*(((r - r_LJ)^2)/2 - (r - r_LJ)) + U_sterics_cut;"
            sterics_energy_expression += "U_sterics_cut = 4*epsilon*((sigma/r_LJ)^6)*(((sigma/r_LJ)^6) - 1.0);"
            sterics_energy_expression += "Force = -4*epsilon*((-12*sigma^12)/(r_LJ^13) + (6*sigma^6)/(r_LJ^7));"
            sterics_energy_expression += "x = (sigma/r)^6;"
            sterics_energy_expression += "r_LJ = softcore_alpha*((26/7)*(sigma^6)*lambda_sterics_deprecated)^(1/6);"
            sterics_energy_expression += "lambda_sterics_deprecated = new_interaction*(1.0 - lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
        else:
            sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"

        return sterics_energy_expression

    @staticmethod
    def _nonbonded_custom_sterics_common():
        """
        Get a custom sterics expression using amber softcore expression

        Returns
        -------
        sterics_addition : str
            The common softcore sterics energy expression

        TODO
        ----
        * Move to a dictionary or equivalent.
        """
        # interpolation
        sterics_addition = "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;"
        # effective softcore distance for sterics
        sterics_addition += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
        sterics_addition += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"

        sterics_addition += "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
        sterics_addition += "lambda_sterics = core_interaction*lambda_sterics_core + new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
        sterics_addition += "core_interaction = delta(unique_old1+unique_old2+unique_new1+unique_new2);new_interaction = max(unique_new1, unique_new2);old_interaction = max(unique_old1, unique_old2);"

        return sterics_addition

    @staticmethod
    def _nonbonded_custom_mixing_rules():
        """
        Mixing rules for the custom nonbonded force.

        Returns
        -------
        sterics_mixing_rules : str
            The mixing expression for sterics
        electrostatics_mixing_rules : str
            The mixiing rules for electrostatics

        TODO
        ----
        * Move to a dictionary or equivalent.
        """
        # Define mixing rules.
        # mixing rule for epsilon
        sterics_mixing_rules = "epsilonA = sqrt(epsilonA1*epsilonA2);"
        # mixing rule for epsilon
        sterics_mixing_rules += "epsilonB = sqrt(epsilonB1*epsilonB2);"
        # mixing rule for sigma
        sterics_mixing_rules += "sigmaA = 0.5*(sigmaA1 + sigmaA2);"
        # mixing rule for sigma
        sterics_mixing_rules += "sigmaB = 0.5*(sigmaB1 + sigmaB2);"
        return sterics_mixing_rules

    @staticmethod
    def _translate_nonbonded_method_to_custom(standard_nonbonded_method):
        """
        Utility function to translate the nonbonded method enum from the
        standard nonbonded force to the custom version
        `CutoffPeriodic`, `PME`, and `Ewald` all become `CutoffPeriodic`;
        `NoCutoff` becomes `NoCutoff`; `CutoffNonPeriodic` becomes
        `CutoffNonPeriodic`

        Parameters
        ----------
        standard_nonbonded_method : openmm.NonbondedForce.NonbondedMethod
            the nonbonded method of the standard force

        Returns
        -------
        custom_nonbonded_method : openmm.CustomNonbondedForce.NonbondedMethod
            the nonbonded method for the equivalent customnonbonded force
        """
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic,
                                         openmm.NonbondedForce.PME,
                                         openmm.NonbondedForce.Ewald]:
            return openmm.CustomNonbondedForce.CutoffPeriodic
        elif standard_nonbonded_method == openmm.NonbondedForce.NoCutoff:
            return openmm.CustomNonbondedForce.NoCutoff
        elif standard_nonbonded_method == openmm.NonbondedForce.CutoffNonPeriodic:
            return openmm.CustomNonbondedForce.CutoffNonPeriodic
        else:
            errmsg = "This nonbonded method is not supported."
            raise NotImplementedError(errmsg)

    def _add_nonbonded_force_terms(self):
        """
        Add the nonbonded force terms to the hybrid system. Note that as with
        the other forces, this method does not add any interactions. It only
        sets up the forces.

        Notes
        -----
        * User defined functions have been removed for now.
        * Argument `add_custom_sterics_force` (default True) has been removed
          for now.

        TODO
        ----
        * Move nonbonded_method defn here to avoid just setting it globally
          and polluting `self`.
        """
        # Add a regular nonbonded force for all interactions that are not
        # changing.
        standard_nonbonded_force = openmm.NonbondedForce()
        self._hybrid_system.addForce(standard_nonbonded_force)
        self._hybrid_system_forces['standard_nonbonded_force'] = standard_nonbonded_force

        # Create a CustomNonbondedForce to handle alchemically interpolated
        # nonbonded parameters.
        # Select functional form based on nonbonded method.
        # TODO: check _nonbonded_custom_ewald and _nonbonded_custom_cutoff
        # since they take arguments that are never used...
        if self._nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            sterics_energy_expression = self._nonbonded_custom(
                self._softcore_LJ_v2)
        elif self._nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic,
                                        openmm.NonbondedForce.CutoffNonPeriodic]:
            epsilon_solvent = self._old_system_forces['NonbondedForce'].getReactionFieldDielectric()
            r_cutoff = self._old_system_forces['NonbondedForce'].getCutoffDistance()
            sterics_energy_expression = self._nonbonded_custom(
                self._softcore_LJ_v2)
            standard_nonbonded_force.setReactionFieldDielectric(
                epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        elif self._nonbonded_method in [openmm.NonbondedForce.PME,
                                        openmm.NonbondedForce.Ewald]:
            [alpha_ewald, nx, ny, nz] = self._old_system_forces['NonbondedForce'].getPMEParameters()
            delta = self._old_system_forces['NonbondedForce'].getEwaldErrorTolerance()
            r_cutoff = self._old_system_forces['NonbondedForce'].getCutoffDistance()
            sterics_energy_expression = self._nonbonded_custom(
                self._softcore_LJ_v2)
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        else:
            errmsg = f"Nonbonded method {self._nonbonded_method} not supported"
            raise ValueError(errmsg)

        standard_nonbonded_force.setNonbondedMethod(self._nonbonded_method)

        sterics_energy_expression += self._nonbonded_custom_sterics_common()

        sterics_mixing_rules = self._nonbonded_custom_mixing_rules()

        custom_nonbonded_method = self._translate_nonbonded_method_to_custom(
            self._nonbonded_method)

        total_sterics_energy = "U_sterics;" + sterics_energy_expression + sterics_mixing_rules

        sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(
            total_sterics_energy)

        if self._softcore_LJ_v2:
            sterics_custom_nonbonded_force.addGlobalParameter(
                "softcore_alpha", self._softcore_LJ_v2_alpha)
        else:
            sterics_custom_nonbonded_force.addGlobalParameter(
                "softcore_alpha", self._softcore_alpha)

        # Lennard-Jones sigma initial
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaA")
        # Lennard-Jones epsilon initial
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonA")
        # Lennard-Jones sigma final
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaB")
        # Lennard-Jones epsilon final
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonB")
        # 1 = hybrid old atom, 0 otherwise
        sterics_custom_nonbonded_force.addPerParticleParameter("unique_old")
        # 1 = hybrid new atom, 0 otherwise
        sterics_custom_nonbonded_force.addPerParticleParameter("unique_new")

        sterics_custom_nonbonded_force.addGlobalParameter(
            "lambda_sterics_core", 0.0)
        sterics_custom_nonbonded_force.addGlobalParameter(
            "lambda_electrostatics_core", 0.0)
        sterics_custom_nonbonded_force.addGlobalParameter(
            "lambda_sterics_insert", 0.0)
        sterics_custom_nonbonded_force.addGlobalParameter(
            "lambda_sterics_delete", 0.0)

        sterics_custom_nonbonded_force.setNonbondedMethod(
            custom_nonbonded_method)

        self._hybrid_system.addForce(sterics_custom_nonbonded_force)
        self._hybrid_system_forces['core_sterics_force'] = sterics_custom_nonbonded_force

        # Set the use of dispersion correction to be the same between the new
        # nonbonded force and the old one:
        if self._old_system_forces['NonbondedForce'].getUseDispersionCorrection():
            self._hybrid_system_forces['standard_nonbonded_force'].setUseDispersionCorrection(True)
            if self._use_dispersion_correction:
                sterics_custom_nonbonded_force.setUseLongRangeCorrection(True)
        else:
            self._hybrid_system_forces['standard_nonbonded_force'].setUseDispersionCorrection(False)

        if self._old_system_forces['NonbondedForce'].getUseSwitchingFunction():
            switching_distance = self._old_system_forces['NonbondedForce'].getSwitchingDistance()
            standard_nonbonded_force.setUseSwitchingFunction(True)
            standard_nonbonded_force.setSwitchingDistance(switching_distance)
            sterics_custom_nonbonded_force.setUseSwitchingFunction(True)
            sterics_custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:
            standard_nonbonded_force.setUseSwitchingFunction(False)
            sterics_custom_nonbonded_force.setUseSwitchingFunction(False)

    @staticmethod
    def _find_bond_parameters(bond_force, index1, index2):
        """
        This is a convenience function to find bond parameters in another
        system given the two indices.

        Parameters
        ----------
        bond_force : openmm.HarmonicBondForce
            The bond force where the parameters should be found
        index1 : int
           Index1 (order does not matter) of the bond atoms
        index2 : int
           Index2 (order does not matter) of the bond atoms

        Returns
        -------
        bond_parameters : list
            List of relevant bond parameters
        """
        index_set = {index1, index2}
        # Loop through all the bonds:
        for bond_index in range(bond_force.getNumBonds()):
            parms = bond_force.getBondParameters(bond_index)
            if index_set == {parms[0], parms[1]}:
                return parms

        return []

    def _handle_harmonic_bonds(self):
        """
        This method adds the appropriate interaction for all bonds in the
        hybrid system. The scheme used is:

        1) If the two atoms are both in the core, then we add to the
            CustomBondForce and interpolate between the two parameters
        2) If one of the atoms is in core and the other is environment, we
           have to assert that the bond parameters do not change between the
           old and the new system; then, the parameters are added to the
           regular bond force
        3) Otherwise, we add the bond to a regular bond force.

        Notes
        -----
        * Bond softening logic has been removed for now.
        """
        old_system_bond_force = self._old_system_forces['HarmonicBondForce']
        new_system_bond_force = self._new_system_forces['HarmonicBondForce']

        # First, loop through the old system bond forces and add relevant terms
        for bond_index in range(old_system_bond_force.getNumBonds()):
            # Get each set of bond parameters
            [index1_old, index2_old, r0_old, k_old] = old_system_bond_force.getBondParameters(bond_index)

            # Map the indices to the hybrid system, for which our atom classes
            # are defined.
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}

            # Now check if it is a subset of the core atoms (that is, both
            # atoms are in the core)
            # If it is, we need to find the parameters in the old system so
            # that we can interpolate
            if index_set.issubset(self._atom_classes['core_atoms']):
                index1_new = self._old_to_new_map[index1_old]
                index2_new = self._old_to_new_map[index2_old]
                new_bond_parameters = self._find_bond_parameters(
                    new_system_bond_force, index1_new, index2_new)
                if not new_bond_parameters:
                    r0_new = r0_old
                    k_new = 0.0*unit.kilojoule_per_mole/unit.angstrom**2
                else:
                    # TODO - why is this being recalculated?
                    [index1, index2, r0_new, k_new] = self._find_bond_parameters(
                        new_system_bond_force, index1_new, index2_new)
                self._hybrid_system_forces['core_bond_force'].addBond(
                    index1_hybrid, index2_hybrid,
                    [r0_old, k_old, r0_new, k_new])

            # Check if the index set is a subset of anything besides
            # environment (in the case of environment, we just add the bond to
            # the regular bond force)
            # that would mean that this bond is core-unique_old or
            # unique_old-unique_old
            # NOTE - These are currently all the same because we don't soften
            # TODO - work these out somewhere else, this is terribly difficult
            #        to understand logic.
            elif (index_set.issubset(self._atom_classes['unique_old_atoms']) or
                  (len(index_set.intersection(self._atom_classes['unique_old_atoms'])) == 1
                   and len(index_set.intersection(self._atom_classes['core_atoms'])) == 1)):

                # We can just add it to the regular bond force.
                self._hybrid_system_forces['standard_bond_force'].addBond(
                    index1_hybrid, index2_hybrid, r0_old, k_old)

            elif (len(index_set.intersection(self._atom_classes['environment_atoms'])) == 1 and
                  len(index_set.intersection(self._atom_classes['core_atoms'])) == 1):
                self._hybrid_system_forces['standard_bond_force'].addBond(
                    index1_hybrid, index2_hybrid, r0_old, k_old)

            # Otherwise, we just add the same parameters as those in the old
            # system (these are environment atoms, and the parameters are the
            # same)
            elif index_set.issubset(self._atom_classes['environment_atoms']):
                self._hybrid_system_forces['standard_bond_force'].addBond(
                    index1_hybrid, index2_hybrid, r0_old, k_old)
            else:
                errmsg = (f"hybrid index set {index_set} does not fit into a "
                          "canonical atom type")
                raise ValueError(errmsg)

        # Now loop through the new system to get the interactions that are
        # unique to it.
        for bond_index in range(new_system_bond_force.getNumBonds()):
            # Get each set of bond parameters
            [index1_new, index2_new, r0_new, k_new] = new_system_bond_force.getBondParameters(bond_index)

            # Convert indices to hybrid, since that is how we represent atom classes:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]
            index_set = {index1_hybrid, index2_hybrid}

            # If the intersection of this set and unique new atoms contains
            # anything, the bond is unique to the new system and must be added
            # all other bonds in the new system have been accounted for already
            # NOTE - These are mostly all the same because we don't soften
            if (len(index_set.intersection(self._atom_classes['unique_new_atoms'])) == 2 or
                (len(index_set.intersection(self._atom_classes['unique_new_atoms'])) == 1 and
                 len(index_set.intersection(self._atom_classes['core_atoms'])) == 1)):

                # If we aren't softening bonds, then just add it to the standard bond force
                self._hybrid_system_forces['standard_bond_force'].addBond(
                    index1_hybrid, index2_hybrid, r0_new, k_new)

            # If the bond is in the core, it has probably already been added
            # in the above loop. However, there are some circumstances
            # where it was not (closing a ring). In that case, the bond has
            # not been added and should be added here.
            # This has some peculiarities to be discussed...
            # TODO - Work out what the above peculiarities are...
            elif index_set.issubset(self._atom_classes['core_atoms']):
                if not self._find_bond_parameters(
                        self._hybrid_system_forces['core_bond_force'],
                        index1_hybrid, index2_hybrid):
                    r0_old = r0_new
                    k_old = 0.0*unit.kilojoule_per_mole/unit.angstrom**2
                    self._hybrid_system_forces['core_bond_force'].addBond(
                        index1_hybrid, index2_hybrid,
                        [r0_old, k_old, r0_new, k_new])
            elif index_set.issubset(self._atom_classes['environment_atoms']):
                # Already been added
                pass

            elif (len(index_set.intersection(self._atom_classes['environment_atoms'])) == 1 and
                  len(index_set.intersection(self._atom_classes['core_atoms'])) == 1):
                pass

            else:
                errmsg = (f"hybrid index set {index_set} does not fit into a "
                          "canonical atom type")
                raise ValueError(errmsg)

    @staticmethod
    def _find_angle_parameters(angle_force, indices):
        """
        Convenience function to find the angle parameters corresponding to a
        particular set of indices

        Parameters
        ----------
        angle_force : openmm.HarmonicAngleForce
            The force where the angle of interest may be found.
        indices : list of int
            The indices (any order) of the angle atoms

        Returns
        -------
        angle_params : list
            list of angle parameters
        """
        indices_reversed = indices[::-1]

        # Now loop through and try to find the angle:
        for angle_index in range(angle_force.getNumAngles()):
            angle_params = angle_force.getAngleParameters(angle_index)

            # Get a set representing the angle indices
            angle_param_indices = angle_params[:3]

            if (indices == angle_param_indices or
                    indices_reversed == angle_param_indices):
                return angle_params
        return []  # Return empty if no matching angle found

    def _handle_harmonic_angles(self):
        """
        This method adds the appropriate interaction for all angles in the
        hybrid system. The scheme used, as with bonds, is:

        1) If the three atoms are all in the core, then we add to the
           CustomAngleForce and interpolate between the two parameters
        2) If the three atoms contain at least one unique new, check if the
           angle is in the neglected new list, and if so, interpolate from
           K_1 = 0; else, if the three atoms contain at least one unique old,
           check if the angle is in the neglected old list, and if so,
           interpolate from K_2 = 0.
        3) If the angle contains at least one environment and at least one
           core atom, assert there are no unique new atoms and that the angle
           terms are preserved between the new and the old system.  Then add to
           the standard angle force.
        4) Otherwise, we add the angle to a regular angle force since it is
           environment.

        Notes
        -----
        * Removed softening and neglected angle functionality
        """
        old_system_angle_force = self._old_system_forces['HarmonicAngleForce']
        new_system_angle_force = self._new_system_forces['HarmonicAngleForce']

        # First, loop through all the angles in the old system to determine
        # what to do with them. We will only use the
        # custom angle force if all atoms are part of "core." Otherwise, they
        # are either unique to one system or never change.
        for angle_index in range(old_system_angle_force.getNumAngles()):

            old_angle_parameters = old_system_angle_force.getAngleParameters(
                                       angle_index)

            # Get the indices in the hybrid system
            hybrid_index_list = [
                self._old_to_hybrid_map[old_atomid] for old_atomid in old_angle_parameters[:3]
            ]
            hybrid_index_set = set(hybrid_index_list)

            # If all atoms are in the core, we'll need to find the
            # corresponding parameters in the old system and interpolate
            if hybrid_index_set.issubset(self._atom_classes['core_atoms']):
                # Get the new indices so we can get the new angle parameters
                new_indices = [
                    self._old_to_new_map[old_atomid] for old_atomid in old_angle_parameters[:3]
                ]
                new_angle_parameters = self._find_angle_parameters(
                    new_system_angle_force, new_indices
                )
                if not new_angle_parameters:
                    new_angle_parameters = [
                        0, 0, 0, old_angle_parameters[3],
                        0.0*unit.kilojoule_per_mole/unit.radian**2
                    ]

                # Add to the hybrid force:
                # the parameters at indices 3 and 4 represent theta0 and k,
                # respectively.
                hybrid_force_parameters = [
                    old_angle_parameters[3], old_angle_parameters[4],
                    new_angle_parameters[3], new_angle_parameters[4]
                ]
                self._hybrid_system_forces['core_angle_force'].addAngle(
                    hybrid_index_list[0], hybrid_index_list[1],
                    hybrid_index_list[2], hybrid_force_parameters
                )

            # Check if the atoms are neither all core nor all environment,
            # which would mean they involve unique old interactions
            elif not hybrid_index_set.issubset(
                    self._atom_classes['environment_atoms']):
                # if there is an environment atom
                if hybrid_index_set.intersection(
                        self._atom_classes['environment_atoms']):
                    if hybrid_index_set.intersection(
                            self._atom_classes['unique_old_atoms']):
                        errmsg = "we disallow unique-environment terms"
                        raise ValueError(errmsg)

                    self._hybrid_system_forces['standard_angle_force'].addAngle(
                        hybrid_index_list[0], hybrid_index_list[1],
                        hybrid_index_list[2], old_angle_parameters[3],
                        old_angle_parameters[4]
                    )
                else:
                    # There are no env atoms, so we can treat this term
                    # appropriately

                    # We don't soften so just add this to the standard angle
                    # force
                    self._hybrid_system_forces['standard_angle_force'].addAngle(
                        hybrid_index_list[0], hybrid_index_list[1],
                        hybrid_index_list[2], old_angle_parameters[3],
                        old_angle_parameters[4]
                    )

            # Otherwise, only environment atoms are in this interaction, so
            # add it to the standard angle force
            elif hybrid_index_set.issubset(
                    self._atom_classes['environment_atoms']):
                self._hybrid_system_forces['standard_angle_force'].addAngle(
                    hybrid_index_list[0], hybrid_index_list[1],
                    hybrid_index_list[2], old_angle_parameters[3],
                    old_angle_parameters[4]
                )
            else:
                errmsg = (f"handle_harmonic_angles: angle_index {angle_index} "
                          "does not fit a canonical form.")
                raise ValueError(errmsg)

        # Finally, loop through the new system force to add any unique new
        # angles
        for angle_index in range(new_system_angle_force.getNumAngles()):

            new_angle_parameters = new_system_angle_force.getAngleParameters(
                                       angle_index)

            # Get the indices in the hybrid system
            hybrid_index_list = [
                self._new_to_hybrid_map[new_atomid] for new_atomid in new_angle_parameters[:3]
            ]
            hybrid_index_set = set(hybrid_index_list)

            # If the intersection of this hybrid set with the unique new atoms
            # is nonempty, it must be added:
            # TODO - there's a ton of len > 0 on sets, empty sets == False,
            #        so we can simplify this logic.
            if len(hybrid_index_set.intersection(
                    self._atom_classes['unique_new_atoms'])) > 0:
                if hybrid_index_set.intersection(
                        self._atom_classes['environment_atoms']):
                    errmsg = ("we disallow angle terms with unique new and "
                              "environment atoms")
                    raise ValueError(errmsg)

                # Not softening just add to the nonalchemical force
                self._hybrid_system_forces['standard_angle_force'].addAngle(
                    hybrid_index_list[0], hybrid_index_list[1],
                    hybrid_index_list[2], new_angle_parameters[3],
                    new_angle_parameters[4]
                )

            elif hybrid_index_set.issubset(self._atom_classes['core_atoms']):
                if not self._find_angle_parameters(self._hybrid_system_forces['core_angle_force'],
                                                   hybrid_index_list):
                    hybrid_force_parameters = [
                        new_angle_parameters[3],
                        0.0*unit.kilojoule_per_mole/unit.radian**2,
                        new_angle_parameters[3], new_angle_parameters[4]
                    ]
                    self._hybrid_system_forces['core_angle_force'].addAngle(
                        hybrid_index_list[0], hybrid_index_list[1],
                        hybrid_index_list[2], hybrid_force_parameters
                    )
            elif hybrid_index_set.issubset(self._atom_classes['environment_atoms']):
                # We have already added the appropriate environmental atom
                # terms
                pass
            elif hybrid_index_set.intersection(self._atom_classes['environment_atoms']):
                if hybrid_index_set.intersection(self._atom_classes['unique_new_atoms']):
                    errmsg = ("we disallow angle terms with unique new and "
                              "environment atoms")
                    raise ValueError(errmsg)
            else:
                errmsg = (f"hybrid index list {hybrid_index_list} does not "
                          "fit into a canonical atom set")
                raise ValueError(errmsg)

    @staticmethod
    def _find_torsion_parameters(torsion_force, indices):
        """
        Convenience function to find the torsion parameters corresponding to a
        particular set of indices.

        Parameters
        ----------
        torsion_force : openmm.PeriodicTorsionForce
            torsion force where the torsion of interest may be found
        indices : list of int
            The indices of the atoms of the torsion

        Returns
        -------
        torsion_parameters : list
            torsion parameters
        """
        indices_reversed = indices[::-1]

        torsion_params_list = list()

        # Now loop through and try to find the torsion:
        for torsion_idx in range(torsion_force.getNumTorsions()):
            torsion_params = torsion_force.getTorsionParameters(torsion_idx)

            # Get a set representing the torsion indices:
            torsion_param_indices = torsion_params[:4]

            if (indices == torsion_param_indices or
                    indices_reversed == torsion_param_indices):
                torsion_params_list.append(torsion_params)

        return torsion_params_list

    def _handle_periodic_torsion_force(self):
        """
        Handle the torsions defined in the new and old systems as such:

        1. old system torsions will enter the ``custom_torsion_force`` if they
           do not contain ``unique_old_atoms`` and will interpolate from ``on``
           to ``off`` from ``lambda_torsions`` = 0 to 1, respectively.
        2. new system torsions will enter the ``custom_torsion_force`` if they
           do not contain ``unique_new_atoms`` and will interpolate from
           ``off`` to ``on`` from ``lambda_torsions`` = 0 to 1, respectively.
        3. old *and* new system torsions will enter the
           ``unique_atom_torsion_force`` (``standard_torsion_force``) and will
           *not* be interpolated.

        Notes
        -----
        * Torsion flattening logic has been removed for now.
        """
        old_system_torsion_force = self._old_system_forces['PeriodicTorsionForce']
        new_system_torsion_force = self._new_system_forces['PeriodicTorsionForce']

        auxiliary_custom_torsion_force = []
        old_custom_torsions_to_standard = []

        # We need to keep track of what torsions we added so that we do not
        # double count
        # added_torsions = []
        # TODO: Commented out since this actually isn't being done anywhere?
        #       Is it necessary? Should we add this logic back in?
        for torsion_index in range(old_system_torsion_force.getNumTorsions()):

            torsion_parameters = old_system_torsion_force.getTorsionParameters(
                                     torsion_index)

            # Get the indices in the hybrid system
            hybrid_index_list = [
                self._old_to_hybrid_map[old_index] for old_index in torsion_parameters[:4]
            ]
            hybrid_index_set = set(hybrid_index_list)

            # If all atoms are in the core, we'll need to find the
            # corresponding parameters in the old system and interpolate
            if hybrid_index_set.intersection(self._atom_classes['unique_old_atoms']):
                # Then it goes to a standard force...
                self._hybrid_system_forces['unique_atom_torsion_force'].addTorsion(
                    hybrid_index_list[0], hybrid_index_list[1],
                    hybrid_index_list[2], hybrid_index_list[3],
                    torsion_parameters[4], torsion_parameters[5],
                    torsion_parameters[6]
                )
            else:
                # It is a core-only term, an environment-only term, or a
                # core/env term; in any case, it goes to the core torsion_force
                # TODO - why are we even adding the 0.0, 0.0, 0.0 section?
                hybrid_force_parameters = [
                    torsion_parameters[4], torsion_parameters[5],
                    torsion_parameters[6], 0.0, 0.0, 0.0
                ]
                auxiliary_custom_torsion_force.append(
                    [hybrid_index_list[0], hybrid_index_list[1],
                     hybrid_index_list[2], hybrid_index_list[3],
                     hybrid_force_parameters[:3]]
                )

        for torsion_index in range(new_system_torsion_force.getNumTorsions()):
            torsion_parameters = new_system_torsion_force.getTorsionParameters(torsion_index)

            # Get the indices in the hybrid system:
            hybrid_index_list = [
                self._new_to_hybrid_map[new_index] for new_index in torsion_parameters[:4]]
            hybrid_index_set = set(hybrid_index_list)

            if hybrid_index_set.intersection(self._atom_classes['unique_new_atoms']):
                # Then it goes to the custom torsion force (scaled to zero)
                self._hybrid_system_forces['unique_atom_torsion_force'].addTorsion(
                    hybrid_index_list[0], hybrid_index_list[1],
                    hybrid_index_list[2], hybrid_index_list[3],
                    torsion_parameters[4], torsion_parameters[5],
                    torsion_parameters[6]
                )
            else:
                hybrid_force_parameters = [
                    0.0, 0.0, 0.0, torsion_parameters[4],
                    torsion_parameters[5], torsion_parameters[6]]

                # Check to see if this term is in the olds...
                term = [hybrid_index_list[0], hybrid_index_list[1],
                        hybrid_index_list[2], hybrid_index_list[3],
                        hybrid_force_parameters[3:]]
                if term in auxiliary_custom_torsion_force:
                    # Then this terms has to go to standard and be deleted...
                    old_index = auxiliary_custom_torsion_force.index(term)
                    old_custom_torsions_to_standard.append(old_index)
                    self._hybrid_system_forces['unique_atom_torsion_force'].addTorsion(
                        hybrid_index_list[0], hybrid_index_list[1],
                        hybrid_index_list[2], hybrid_index_list[3],
                        torsion_parameters[4], torsion_parameters[5],
                        torsion_parameters[6]
                    )
                else:
                    # Then this term has to go to the core force...
                    self._hybrid_system_forces['custom_torsion_force'].addTorsion(
                        hybrid_index_list[0], hybrid_index_list[1],
                        hybrid_index_list[2], hybrid_index_list[3],
                        hybrid_force_parameters
                    )

        # Now we have to loop through the aux custom torsion force
        for index in [q for q in range(len(auxiliary_custom_torsion_force))
                      if q not in old_custom_torsions_to_standard]:
            terms = auxiliary_custom_torsion_force[index]
            hybrid_index_list = terms[:4]
            hybrid_force_parameters = terms[4] + [0., 0., 0.]
            self._hybrid_system_forces['custom_torsion_force'].addTorsion(
                hybrid_index_list[0], hybrid_index_list[1],
                hybrid_index_list[2], hybrid_index_list[3],
                hybrid_force_parameters
            )

    def _handle_nonbonded(self):
        """
        Handle the nonbonded interactions defined in the new and old systems.

        TODO
        ----
        * Expand this docstring to explain the logic.
        * A lot of this logic is duplicated, probably turn it into a couple of
          functions.
        """
        def _check_indices(idx1, idx2):
            if idx1 != idx2:
                errmsg = ("Attempting to add incorrect particle to hybrid "
                          "system")
                raise ValueError(errmsg)

        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = self._hybrid_to_old_map
        hybrid_to_new_map = self._hybrid_to_new_map

        # Define new global parameters for NonbondedForce
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter('lambda_electrostatics_core', 0.0)
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter('lambda_sterics_core', 0.0)
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter("lambda_electrostatics_delete", 0.0)
        self._hybrid_system_forces['standard_nonbonded_force'].addGlobalParameter("lambda_electrostatics_insert", 0.0)

        # We have to loop through the particles in the system, because
        # nonbonded force does not accept index
        for particle_index in range(self._hybrid_system.getNumParticles()):

            if particle_index in self._atom_classes['unique_old_atoms']:
                # Get the parameters in the old system
                old_index = hybrid_to_old_map[particle_index]
                [charge, sigma, epsilon] = old_system_nonbonded_force.getParticleParameters(old_index)

                # Add the particle to the hybrid custom sterics and
                # electrostatics.
                # turning off sterics in forward direction
                check_index = self._hybrid_system_forces['core_sterics_force'].addParticle(
                    [sigma, epsilon, sigma, 0.0*epsilon, 1, 0]
                )
                _check_indices(particle_index, check_index)

                # Add particle to the regular nonbonded force, but
                # Lennard-Jones will be handled by CustomNonbondedForce
                check_index = self._hybrid_system_forces['standard_nonbonded_force'].addParticle(
                    charge, sigma, 0.0*epsilon
                )
                _check_indices(particle_index, check_index)

                # Charge will be turned off at
                # lambda_electrostatics_delete = 0, on at
                # lambda_electrostatics_delete = 1; kill charge with
                # lambda_electrostatics_delete = 0 --> 1
                self._hybrid_system_forces['standard_nonbonded_force'].addParticleParameterOffset(
                    'lambda_electrostatics_delete', particle_index,
                    -charge, 0*sigma, 0*epsilon
                )

            elif particle_index in self._atom_classes['unique_new_atoms']:
                # Get the parameters in the new system
                new_index = hybrid_to_new_map[particle_index]
                [charge, sigma, epsilon] = new_system_nonbonded_force.getParticleParameters(new_index)

                # Add the particle to the hybrid custom sterics and electrostatics
                # turning on sterics in forward direction
                check_index = self._hybrid_system_forces['core_sterics_force'].addParticle(
                    [sigma, 0.0*epsilon, sigma, epsilon, 0, 1]
                )
                _check_indices(particle_index, check_index)

                # Add particle to the regular nonbonded force, but
                # Lennard-Jones will be handled by CustomNonbondedForce
                check_index = self._hybrid_system_forces['standard_nonbonded_force'].addParticle(
                    0.0, sigma, 0.0
                )  # charge starts at zero
                _check_indices(particle_index, check_index)

                # Charge will be turned off at lambda_electrostatics_insert = 0
                # on at lambda_electrostatics_insert = 1;
                # add charge with lambda_electrostatics_insert = 0 --> 1
                self._hybrid_system_forces['standard_nonbonded_force'].addParticleParameterOffset(
                    'lambda_electrostatics_insert', particle_index,
                    +charge, 0, 0
                )

            elif particle_index in self._atom_classes['core_atoms']:
                # Get the parameters in the new and old systems:
                old_index = hybrid_to_old_map[particle_index]
                [charge_old, sigma_old, epsilon_old] = old_system_nonbonded_force.getParticleParameters(old_index)
                new_index = hybrid_to_new_map[particle_index]
                [charge_new, sigma_new, epsilon_new] = new_system_nonbonded_force.getParticleParameters(new_index)

                # Add the particle to the custom forces, interpolating between
                # the two parameters; add steric params and zero electrostatics
                # to core_sterics per usual
                check_index = self._hybrid_system_forces['core_sterics_force'].addParticle(
                    [sigma_old, epsilon_old, sigma_new, epsilon_new, 0, 0])
                _check_indices(particle_index, check_index)

                # Still add the particle to the regular nonbonded force, but
                # with zeroed out parameters; add old charge to
                # standard_nonbonded and zero sterics
                check_index = self._hybrid_system_forces['standard_nonbonded_force'].addParticle(
                    charge_old, 0.5*(sigma_old+sigma_new), 0.0)
                _check_indices(particle_index, check_index)

                # Charge is charge_old at lambda_electrostatics = 0,
                # charge_new at lambda_electrostatics = 1
                # TODO: We could also interpolate the Lennard-Jones here
                # instead of core_sterics force so that core_sterics_force
                # could just be softcore.

                # Interpolate between old and new charge with
                # lambda_electrostatics core make sure to keep sterics off
                self._hybrid_system_forces['standard_nonbonded_force'].addParticleParameterOffset(
                    'lambda_electrostatics_core', particle_index,
                    (charge_new - charge_old), 0, 0
                )

            # Otherwise, the particle is in the environment
            else:
                # The parameters will be the same in new and old system, so
                # just take the old parameters
                old_index = hybrid_to_old_map[particle_index]
                [charge, sigma, epsilon] = old_system_nonbonded_force.getParticleParameters(old_index)

                # Add the particle to the hybrid custom sterics, but they dont
                # change; electrostatics are ignored
                self._hybrid_system_forces['core_sterics_force'].addParticle(
                    [sigma, epsilon, sigma, epsilon, 0, 0]
                )

                # Add the environment atoms to the regular nonbonded force as
                # well: should we be adding steric terms here, too?
                self._hybrid_system_forces['standard_nonbonded_force'].addParticle(
                    charge, sigma, epsilon
                )

        # Now loop pairwise through (unique_old, unique_new) and add exceptions
        # so that they never interact electrostatically
        # (place into Nonbonded Force)
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']

        for old in unique_old_atoms:
            for new in unique_new_atoms:
                self._hybrid_system_forces['standard_nonbonded_force'].addException(
                    old, new, 0.0*unit.elementary_charge**2,
                    1.0*unit.nanometers, 0.0*unit.kilojoules_per_mole)
                # This is only necessary to avoid the 'All forces must have
                # identical exclusions' rule
                self._hybrid_system_forces['core_sterics_force'].addExclusion(old, new)

        self._handle_interaction_groups()

        self._handle_hybrid_exceptions()

        self._handle_original_exceptions()

    def _handle_interaction_groups(self):
        """
        Create the appropriate interaction groups for the custom nonbonded
        forces. The groups are:

        1) Unique-old - core
        2) Unique-old - environment
        3) Unique-new - core
        4) Unique-new - environment
        5) Core - environment
        6) Core - core

        Unique-old and Unique new are prevented from interacting this way,
        and intra-unique interactions occur in an unmodified nonbonded force.

        Must be called after particles are added to the Nonbonded forces
        TODO: we should also be adding the following interaction groups...
        7) Unique-new - Unique-new
        8) Unique-old - Unique-old
        """
        # Get the force objects for convenience:
        sterics_custom_force = self._hybrid_system_forces['core_sterics_force']

        # Also prepare the atom classes
        core_atoms = self._atom_classes['core_atoms']
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']
        environment_atoms = self._atom_classes['environment_atoms']

        sterics_custom_force.addInteractionGroup(unique_old_atoms, core_atoms)

        sterics_custom_force.addInteractionGroup(unique_old_atoms,
                                                 environment_atoms)

        sterics_custom_force.addInteractionGroup(unique_new_atoms,
                                                 core_atoms)

        sterics_custom_force.addInteractionGroup(unique_new_atoms,
                                                 environment_atoms)

        sterics_custom_force.addInteractionGroup(core_atoms, environment_atoms)

        sterics_custom_force.addInteractionGroup(core_atoms, core_atoms)

        sterics_custom_force.addInteractionGroup(unique_new_atoms,
                                                 unique_new_atoms)

        sterics_custom_force.addInteractionGroup(unique_old_atoms,
                                                 unique_old_atoms)

    def _handle_hybrid_exceptions(self):
        """
        Instead of excluding interactions that shouldn't occur, we provide
        exceptions for interactions that were zeroed out but should occur.
        """
        # TODO - are these actually used anywhere? Flake8 says no
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']

        # Prepare the atom classes
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']

        # Get the list of interaction pairs for which we need to set exceptions
        unique_old_pairs = list(itertools.combinations(unique_old_atoms, 2))
        unique_new_pairs = list(itertools.combinations(unique_new_atoms, 2))

        # Add back the interactions of the old unique atoms, unless there are
        # exceptions
        for atom_pair in unique_old_pairs:
            # Since the pairs are indexed in the dictionary by the old system
            # indices, we need to convert
            old_index_atom_pair = (self._hybrid_to_old_map[atom_pair[0]],
                                   self._hybrid_to_old_map[atom_pair[1]])

            # Now we check if the pair is in the exception dictionary
            if old_index_atom_pair in self._old_system_exceptions:
                [chargeProd, sigma, epsilon] = self._old_system_exceptions[old_index_atom_pair]
                # if we are interpolating 1,4 exceptions then we have to
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd*0.0,
                        sigma, epsilon*0.0
                    )
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon
                    )

                # Add exclusion to ensure exceptions are consistent
                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    atom_pair[0], atom_pair[1]
                )

            # Check if the pair is in the reverse order and use that if so
            elif old_index_atom_pair[::-1] in self._old_system_exceptions:
                [chargeProd, sigma, epsilon] = self._old_system_exceptions[old_index_atom_pair[::-1]]
                # If we are interpolating 1,4 exceptions then we have to
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd*0.0,
                        sigma, epsilon*0.0
                    )
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

                # Add exclusion to ensure exceptions are consistent
                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    atom_pair[0], atom_pair[1])

            # TODO: work out why there's a bunch of commented out code here
            # Exerpt:
            # If it's not handled by an exception in the original system, we
            # just add the regular parameters as an exception
            # TODO: this implies that the old-old nonbonded interactions (those
            # which are not exceptions) are always self-interacting throughout
            # lambda protocol...

        # Add back the interactions of the new unique atoms, unless there are
        # exceptions
        for atom_pair in unique_new_pairs:
            # Since the pairs are indexed in the dictionary by the new system
            # indices, we need to convert
            new_index_atom_pair = (self._hybrid_to_new_map[atom_pair[0]],
                                   self._hybrid_to_new_map[atom_pair[1]])

            # Now we check if the pair is in the exception dictionary
            if new_index_atom_pair in self._new_system_exceptions:
                [chargeProd, sigma, epsilon] = self._new_system_exceptions[new_index_atom_pair]
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd*0.0,
                        sigma, epsilon*0.0
                    )
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon
                    )

                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    atom_pair[0], atom_pair[1]
                )

            # Check if the pair is present in the reverse order and use that if so
            elif new_index_atom_pair[::-1] in self._new_system_exceptions:
                [chargeProd, sigma, epsilon] = self._new_system_exceptions[new_index_atom_pair[::-1]]
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd*0.0,
                        sigma, epsilon*0.0
                    )
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon
                    )

                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    atom_pair[0], atom_pair[1]
                )


            # TODO: work out why there's a bunch of commented out code here
            # If it's not handled by an exception in the original system, we
            # just add the regular parameters as an exception

    @staticmethod
    def _find_exception(force, index1, index2):
        """
        Find the exception that corresponds to the given indices in the given
        system

        Parameters
        ----------
        force : openmm.NonbondedForce object
            System containing the exceptions
        index1 : int
            The index of the first atom (order is unimportant)
        index2 : int
            The index of the second atom (order is unimportant)

        Returns
        -------
        exception_parameters : list
            List of exception parameters
        """
        index_set = {index1, index2}

        # Loop through the exceptions and try to find one matching the criteria
        for exception_idx in range(force.getNumExceptions()):
            exception_parameters = force.getExceptionParameters(exception_idx)
            if index_set==set(exception_parameters[:2]):
                return exception_parameters
        return []

    def _handle_original_exceptions(self):
        """
        This method ensures that exceptions present in the original systems are
        present in the hybrid appropriately.
        """
        # Get what we need to find the exceptions from the new and old systems:
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = self._hybrid_to_old_map
        hybrid_to_new_map = self._hybrid_to_new_map

        # First, loop through the old system's exceptions and add them to the
        # hybrid appropriately:
        for exception_pair, exception_parameters in self._old_system_exceptions.items():

            [index1_old, index2_old] = exception_pair
            [chargeProd_old, sigma_old, epsilon_old] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}


            # In this case, the interaction is only covered by the regular
            # nonbonded force, and as such will be copied to that force
            # In the unique-old case, it is handled elsewhere due to internal
            # peculiarities regarding exceptions
            if index_set.issubset(self._atom_classes['environment_atoms']):
                self._hybrid_system_forces['standard_nonbonded_force'].addException(
                    index1_hybrid, index2_hybrid, chargeProd_old,
                    sigma_old, epsilon_old
                )
                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    index1_hybrid, index2_hybrid
                )

            # We have already handled unique old - unique old exceptions
            elif len(index_set.intersection(self._atom_classes['unique_old_atoms'])) == 2:
                continue

            # Otherwise, check if one of the atoms in the set is in the
            # unique_old_group and the other is not:
            elif len(index_set.intersection(self._atom_classes['unique_old_atoms'])) == 1:
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        index1_hybrid, index2_hybrid, chargeProd_old*0.0,
                        sigma_old, epsilon_old*0.0
                    )
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        index1_hybrid, index2_hybrid, chargeProd_old,
                        sigma_old, epsilon_old
                    )

                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    index1_hybrid, index2_hybrid
                )

            # If the exception particles are neither solely old unique, solely
            # environment, nor contain any unique old atoms, they are either
            # core/environment or core/core
            # In this case, we need to get the parameters from the exception in
            # the other (new) system, and interpolate between the two
            else:
                # First get the new indices.
                index1_new = hybrid_to_new_map[index1_hybrid]
                index2_new = hybrid_to_new_map[index2_hybrid]
                # Get the exception parameters:
                new_exception_parms= self._find_exception(
                                         new_system_nonbonded_force,
                                         index1_new, index2_new)

                # If there's no new exception, then we should just set the
                # exception parameters to be the nonbonded parameters
                if not new_exception_parms:
                    [charge1_new, sigma1_new, epsilon1_new] = new_system_nonbonded_force.getParticleParameters(index1_new)
                    [charge2_new, sigma2_new, epsilon2_new] = new_system_nonbonded_force.getParticleParameters(index2_new)

                    chargeProd_new = charge1_new * charge2_new
                    sigma_new = 0.5 * (sigma1_new + sigma2_new)
                    epsilon_new = unit.sqrt(epsilon1_new*epsilon2_new)
                else:
                    [index1_new, index2_new, chargeProd_new, sigma_new, epsilon_new] = new_exception_parms

                # Interpolate between old and new
                exception_index = self._hybrid_system_forces['standard_nonbonded_force'].addException(
                    index1_hybrid, index2_hybrid, chargeProd_old,
                    sigma_old, epsilon_old
                )
                self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset(
                    'lambda_electrostatics_core', exception_index,
                    (chargeProd_new - chargeProd_old), 0, 0
                )
                self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset(
                    'lambda_sterics_core', exception_index, 0,
                    (sigma_new - sigma_old), (epsilon_new - epsilon_old)
                )
                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    index1_hybrid, index2_hybrid
                )

        # Now, loop through the new system to collect remaining interactions.
        # The only that remain here are uniquenew-uniquenew, uniquenew-core,
        # and uniquenew-environment. There might also be core-core, since not
        # all core-core exceptions exist in both
        for exception_pair, exception_parameters in self._new_system_exceptions.items():
            [index1_new, index2_new] = exception_pair
            [chargeProd_new, sigma_new, epsilon_new] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]

            index_set = {index1_hybrid, index2_hybrid}

            # If it's a subset of unique_new_atoms, then this is an
            # intra-unique interaction and should have its exceptions
            # specified in the regular nonbonded force. However, this is
            # handled elsewhere as above due to pecularities with exception
            # handling
            if index_set.issubset(self._atom_classes['unique_new_atoms']):
                continue

            # Look for the final class- interactions between uniquenew-core and
            # uniquenew-environment. They are treated similarly: they are
            # simply on and constant the entire time (as a valence term)
            elif len(index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                if self._interpolate_14s:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        index1_hybrid, index2_hybrid, chargeProd_new*0.0,
                        sigma_new, epsilon_new*0.0
                    )
                else:
                    self._hybrid_system_forces['standard_nonbonded_force'].addException(
                        index1_hybrid, index2_hybrid, chargeProd_new,
                        sigma_new, epsilon_new
                    )

                self._hybrid_system_forces['core_sterics_force'].addExclusion(
                    index1_hybrid, index2_hybrid
                )

            # However, there may be a core exception that exists in one system
            # but not the other (ring closure)
            elif index_set.issubset(self._atom_classes['core_atoms']):

                # Get the old indices
                try:
                    index1_old = self._new_to_old_map[index1_new]
                    index2_old = self._new_to_old_map[index2_new]
                except KeyError:
                    continue

                # See if it's also in the old nonbonded force. if it is, then we don't need to add it.
                # But if it's not, we need to interpolate
                if not self._find_exception(old_system_nonbonded_force, index1_old, index2_old):

                    [charge1_old, sigma1_old, epsilon1_old] = old_system_nonbonded_force.getParticleParameters(index1_old)
                    [charge2_old, sigma2_old, epsilon2_old] = old_system_nonbonded_force.getParticleParameters(index2_old)

                    chargeProd_old = charge1_old*charge2_old
                    sigma_old = 0.5 * (sigma1_old + sigma2_old)
                    epsilon_old = unit.sqrt(epsilon1_old*epsilon2_old)

                    exception_index = self._hybrid_system_forces['standard_nonbonded_force'].addException(
                                          index1_hybrid, index2_hybrid,
                                          chargeProd_old, sigma_old, 
                                          epsilon_old)

                    self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset(
                        'lambda_electrostatics_core', exception_index,
                        (chargeProd_new - chargeProd_old), 0, 0
                    )

                    self._hybrid_system_forces['standard_nonbonded_force'].addExceptionParameterOffset(
                        'lambda_sterics_core', exception_index, 0,
                        (sigma_new - sigma_old), (epsilon_new - epsilon_old)
                    )

                    self._hybrid_system_forces['core_sterics_force'].addExclusion(
                        index1_hybrid, index2_hybrid
                    )

    def _handle_old_new_exceptions(self):
        """
        Find the exceptions associated with old-old and old-core interactions,
        as well as new-new and new-core interactions.  Theses exceptions will
        be placed in CustomBondedForce that will interpolate electrostatics and
        a softcore potential.

        TODO
        ----
        * Move old_new_bond_exceptions to a dictionary or similar.
        """

        old_new_nonbonded_exceptions = "U_electrostatics + U_sterics;"

        if self._softcore_LJ_v2:
            old_new_nonbonded_exceptions += "U_sterics = select(step(r - r_LJ), 4*epsilon*x*(x-1.0), U_sterics_quad);"
            old_new_nonbonded_exceptions += f"U_sterics_quad = Force*(((r - r_LJ)^2)/2 - (r - r_LJ)) + U_sterics_cut;"
            old_new_nonbonded_exceptions += f"U_sterics_cut = 4*epsilon*((sigma/r_LJ)^6)*(((sigma/r_LJ)^6) - 1.0);"
            old_new_nonbonded_exceptions += f"Force = -4*epsilon*((-12*sigma^12)/(r_LJ^13) + (6*sigma^6)/(r_LJ^7));"
            old_new_nonbonded_exceptions += f"x = (sigma/r)^6;"
            old_new_nonbonded_exceptions += f"r_LJ = softcore_alpha*((26/7)*(sigma^6)*lambda_sterics_deprecated)^(1/6);"
            old_new_nonbonded_exceptions += f"lambda_sterics_deprecated = new_interaction*(1.0 - lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"
        else:
            old_new_nonbonded_exceptions += "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
            old_new_nonbonded_exceptions += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);"
            old_new_nonbonded_exceptions += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
            old_new_nonbonded_exceptions += "lambda_alpha = new_interaction*(1-lambda_sterics_insert) + old_interaction*lambda_sterics_delete;"

        old_new_nonbonded_exceptions += "U_electrostatics = (lambda_electrostatics_insert * unique_new + unique_old * (1 - lambda_electrostatics_delete)) * ONE_4PI_EPS0*chargeProd/r;"
        old_new_nonbonded_exceptions += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0

        old_new_nonbonded_exceptions += "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;" # interpolation
        old_new_nonbonded_exceptions += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"

        old_new_nonbonded_exceptions += "lambda_sterics = new_interaction*lambda_sterics_insert + old_interaction*lambda_sterics_delete;"
        old_new_nonbonded_exceptions += "new_interaction = delta(1-unique_new); old_interaction = delta(1-unique_old);"


        nonbonded_exceptions_force = openmm.CustomBondForce(
                                         old_new_nonbonded_exceptions)
        self._hybrid_system.addForce(nonbonded_exceptions_force)

        # For reference, set name in force dict
        self._hybrid_system_forces['old_new_exceptions_force'] = nonbonded_exceptions_force

        if self._softcore_LJ_v2:
            nonbonded_exceptions_force.addGlobalParameter(
                "softcore_alpha", self._softcore_LJ_v2_alpha
            )
        else:
            nonbonded_exceptions_force.addGlobalParameter(
                "softcore_alpha", self._softcore_alpha
            )

        # electrostatics insert
        nonbonded_exceptions_force.addGlobalParameter(
            "lambda_electrostatics_insert", 0.0
        )
        # electrostatics delete
        nonbonded_exceptions_force.addGlobalParameter(
            "lambda_electrostatics_delete", 0.0
        )
        # sterics insert
        nonbonded_exceptions_force.addGlobalParameter(
            "lambda_sterics_insert", 0.0
        )
        # steric delete
        nonbonded_exceptions_force.addGlobalParameter(
            "lambda_sterics_delete", 0.0
        )

        for parameter in ['chargeProd','sigmaA', 'epsilonA', 'sigmaB',
                          'epsilonB', 'unique_old', 'unique_new']:
            nonbonded_exceptions_force.addPerBondParameter(parameter)

        # Prepare for exceptions loop by grabbing nonbonded forces,
        # hybrid_to_old/new maps
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = self._hybrid_to_old_map
        hybrid_to_new_map = self._hybrid_to_new_map

        # First, loop through the old system's exceptions and add them to the
        # hybrid appropriately:
        for exception_pair, exception_parameters in self._old_system_exceptions.items():

            [index1_old, index2_old] = exception_pair
            [chargeProd_old, sigma_old, epsilon_old] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}

            # Otherwise, check if one of the atoms in the set is in the
            # unique_old_group and the other is not:
            if (len(index_set.intersection(self._atom_classes['unique_old_atoms'])) > 0 and
                (chargeProd_old.value_in_unit_system(unit.md_unit_system) != 0.0 or
                 epsilon_old.value_in_unit_system(unit.md_unit_system) != 0.0)):
                if self._interpolate_14s:
                    # If we are interpolating 1,4s, then we anneal this term
                    # off; otherwise, the exception force is constant and
                    # already handled in the standard nonbonded force
                    nonbonded_exceptions_force.addBond(
                        index1_hybrid, index2_hybrid,
                        [chargeProd_old, sigma_old, epsilon_old, sigma_old,
                         epsilon_old*0.0, 1, 0]
                    )



        # Next, loop through the new system's exceptions and add them to the
        # hybrid appropriately
        for exception_pair, exception_parameters in self._new_system_exceptions.items():
            [index1_new, index2_new] = exception_pair
            [chargeProd_new, sigma_new, epsilon_new] = exception_parameters

            # Get hybrid indices:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]

            index_set = {index1_hybrid, index2_hybrid}

            # Look for the final class- interactions between uniquenew-core and
            # uniquenew-environment. They are treated
            # similarly: they are simply on and constant the entire time
            # (as a valence term)
            if (len(index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0 and
                (chargeProd_new.value_in_unit_system(unit.md_unit_system) != 0.0 or
                 epsilon_new.value_in_unit_system(unit.md_unit_system) != 0.0)):
                if self._interpolate_14s:
                    # If we are interpolating 1,4s, then we anneal this term
                    # on; otherwise, the exception force is constant and
                    # already handled in the standard nonbonded force
                    nonbonded_exceptions_force.addBond(
                        index1_hybrid, index2_hybrid,
                        [chargeProd_new, sigma_new, epsilon_new*0.0,
                         sigma_new, epsilon_new, 0, 1]
                    )

    def _impose_virtual_bonds(self):
        """
        Add a virtual bond between protein subunits and ligand(s) to ensure
        that everything is imaged together.

        TODO
        ----
        * Eventually this needs to be moved to a "restraint" class, even
          though this is a zero bond force.
        * Move away from proteins and also set this for non-protein hosts
        """
        core_atoms = [int(idx) for idx in self._atom_classes['core_atoms']]
        heavy_atoms = [int(idx) for idx in self._hybrid_topology.select('mass > 1.5')]
        core_heavy_atoms = [int(idx) for idx in set(core_atoms).intersection(set(heavy_atoms))]

        # get protein CA atoms
        protein_atoms = [int(idx) for idx in self._hybrid_topology.select('protein and name CA')]

        if len(core_heavy_atoms) == 0 or len(protein_atoms) == 0:
            # nothing to see here, return
            logger.info("no core or protein atoms - no virt bond added")
            return

        if len(set(core_atoms).intersection(set(protein_atoms))) != 0:
            # core atoms are in the protein, return
            logger.info("all core atoms are in protein - no virt bond added")
            return

        cutoff = 1.2  # 1.2 A
        trajectory = mdt.Trajectory([self.hybrid_positions / unit.nanometers],
                                    topology=self._hybrid_topology)
        matches = mdt.compute_neighbors(trajectory, cutoff, core_heavy_atoms,
                                        haystack_indices=protein_atoms,
                                        periodic=True)

        protein_atoms = set()

        for match in matches:
            for index in match:
                protein_atoms.add(int(index))

        protein_atoms = [int(idx) for idx in protein_atoms]

        if not protein_atoms:
            raise ValueError("no matching atoms for virtual bond")

        # Add virtual bond between a core and protein atom to ensure they are
        # periodically replicated together
        bondforce = openmm.CustomBondForce('0')
        bondforce.addBond(core_heavy_atoms[0], protein_atoms[0], [])
        lmsg = (f"Adding virt bond between {core_heavy_atoms[0]} "
                f"and {protein_atoms[0]}")
        logger.info(lmsg)
        self._hybrid_system.addForce(bondforce)

        # Extract protein and molecule chains and indices before adding solvent
        mdtop = trajectory.top
        protein_atom_indices = mdtop.select('protein and (mass > 1)')
        molecule_atom_indices = mdtop.select(
            '(not protein) and (not water) and (mass > 1)')
        protein_chainids = list(set(
            [atom.residue.chain.index for atom in mdtop.atoms
             if atom.index in protein_atom_indices]))
        n_protein_chains = len(protein_chainids)
        protein_chain_atom_indices = dict()

        for chainid in protein_chainids:
            protein_chain_atom_indices[chainid] = mdtop.select(f'protein and chainid {chainid}')

        # Add virtual bond between protein chains so they are imaged together
        if (n_protein_chains > 1):
            chainid = protein_chainids[0]
            iatom = protein_chain_atom_indices[chainid][0]
            for chainid in protein_chainids[1:]:
                jatom = protein_chain_atom_indices[chainid][0]
                lmsg = (f"Adding inter-chain virt bond between atoms {iatom} "
                        f"and {jatom}")
                logger.info(lmsg)
                bondforce.addBond(int(iatom), int(jatom), [])

    def _compute_hybrid_positions(self):
        """
        The positions of the hybrid system. Dimensionality is (n_environment +
        n_core + n_old_unique + n_new_unique),
        The positions are assigned by first copying all the mapped positions
        from the old system in, then copying the
        mapped positions from the new system. This means that there is an
        assumption that the positions common to old and new are the same
        (which is the case for perses as-is).

        Returns
        -------
        hybrid_positions : np.ndarray [n, 3]
            Positions of the hybrid system, in nm
        """
        # Get unitless positions
        old_pos_without_units = np.array(
            self._old_positions.value_in_unit(unit.nanometer))
        new_pos_without_units = np.array(
            self._new_positions.value_in_unit(unit.nanometer))

        # Determine the number of particles in the system
        n_atoms_hybrid = self._hybrid_system.getNumParticles()

        # Initialize an array for hybrid positions
        hybrid_pos_array = np.zeros([n_atoms_hybrid, 3])

        # Loop through the old system indices, and assign positions.
        for old_idx, hybrid_idx in self._old_to_hybrid_map.items():
            hybrid_pos_array[hybrid_idx, :] = old_pos_without_units[old_idx, :]

        # Do the same for new indices. Note that this overwrites some
        # coordinates, but as stated above, the assumption is that these are
        # the same.
        for new_idx, hybrid_idx in self._new_to_hybrid_map.items():
            hybrid_pos_array[hybrid_idx, :] = new_pos_without_units[new_idx, :]

        return unit.Quantity(hybrid_pos_array, unit=unit.nanometers)

    def _create_mdtraj_topology(self):
        """
        Create an MDTraj trajectory of the hybrid system.

        Note
        ----
        This is purely for writing out trajectories and is not expected to be
        parametrized.

        TODO
        ----
        * A lot of this can be simplified / reworked.
        """
        old_top = mdt.Topology.from_openmm(self._old_topology)
        new_top = mdt.Topology.from_openmm(self._new_topology)

        hybrid_topology = copy.deepcopy(old_top)

        added_atoms = dict()

        # Get the core atoms in the new index system (as opposed to the hybrid
        # index system). We will need this later
        core_atoms_new_indices = set(self._core_old_to_new_map.values())

        # Now, add each unique new atom to the topology (this is the same order
        # as the system)
        for particle_idx in self._unique_new_atoms:
            new_particle_hybrid_idx = self._new_to_hybrid_map[particle_idx]
            new_system_atom = new_top.atom(particle_idx)

            # First, we get the residue in the new system associated with this
            # atom
            new_system_res = new_system_atom.residue

            # Next, we have to enumerate the other atoms in that residue to
            # find mapped atoms
            new_system_atom_set = {atom.index for atom in new_system_res.atoms}

            # Now, we find the subset of atoms that are mapped. These must be 
            # in the "core" category, since they are mapped and part of a
            # changing residue
            mapped_new_atom_indices = core_atoms_new_indices.intersection(
                                          new_system_atom_set)

            # Now get the old indices of the above atoms so that we can find
            # the appropriate residue in the old system for this we can use the
            # new to old atom map
            mapped_old_atom_indices = [self._new_to_old_map[atom_idx] for
                                       atom_idx in mapped_new_atom_indices]

            # We can just take the first one--they all have the same residue
            first_mapped_old_atom_index = mapped_old_atom_indices[0]

            # Get the atom object corresponding to this index from the hybrid
            # (which is a deepcopy of the old)
            mapped_hybrid_system_atom = hybrid_topology.atom(
                                            first_mapped_old_atom_index)

            # Get the residue that is relevant to this atom
            mapped_residue = mapped_hybrid_system_atom.residue

            # Add the atom using the mapped residue
            added_atoms[new_particle_hybrid_idx] = hybrid_topology.add_atom(
                                                       new_system_atom.name,
                                                       new_system_atom.element,
                                                       mapped_residue)

        # Now loop through the bonds in the new system, and if the bond
        # contains a unique new atom, then add it to the hybrid topology
        for (atom1, atom2) in new_top.bonds:
            at1_hybrid_idx = self._new_to_hybrid_map[atom1.index]
            at2_hybrid_idx = self._new_to_hybrid_map[atom2.index]

            # If at least one atom is in the unique new class, we need to add
            # it to the hybrid system
            at1_uniq = at1_hybrid_idx in self._atom_classes['unique_new_atoms']
            at2_uniq = at2_hybrid_idx in self._atom_classes['unique_new_atoms']
            if at1_uniq or at2_uniq:
                if atom1.index in self._atom_classes['unique_new_atoms']:
                    atom1_to_bond = added_atoms[atom1.index]
                else:
                    atom1_to_bond = atom1

                if atom2.index in self._atom_classes['unique_new_atoms']:
                    atom2_to_bond = added_atoms[atom2.index]
                else:
                    atom2_to_bond = atom2

                hybrid_topology.add_bond(atom1_to_bond, atom2_to_bond)

        return hybrid_topology


    def old_positions(self, hybrid_positions):
        """
        From input hybrid positions, get the positions which would correspond
        to the old system

        Parameters
        ----------
        hybrid_positions : [n, 3] np.ndarray or simtk.unit.Quantity
            The positions of the hybrid system

        Returns
        -------
        old_positions : [m, 3] np.ndarray with unit
            The positions of the old system
        """
        n_atoms_old = self._old_system.getNumParticles()
        # making sure hybrid positions are simtk.unit.Quantity objects
        if not isinstance(hybrid_positions, unit.Quantity):
            hybrid_positions = unit.Quantity(hybrid_positions,
                                             unit=unit.nanometer)
        old_positions = unit.Quantity(np.zeros([n_atoms_old, 3]),
                                      unit=unit.nanometer)
        for idx in range(n_atoms_old):
            hyb_idx = self._new_to_hybrid_map[idx]
            old_positions[idx, :] = hybrid_positions[hyb_idx, :]
        return old_positions

    def new_positions(self, hybrid_positions):
        """
        From input hybrid positions, get the positions which could correspond
        to the new system.

        Parameters
        ----------
        hybrid_positions : [n, 3] np.ndarray or simtk.unit.Quantity
            The positions of the hybrid system

        Returns
        -------
        new_positions : [m, 3] np.ndarray with unit
            The positions of the new system
        """
        n_atoms_new = self._new_system.getNumParticles
        # making sure hybrid positions are simtk.unit.Quantity objects
        if not isinstance(hybrid_positions, unit.Quantity):
            hybrid_positions = unit.Quantity(hybrid_positions,
                                             unit=unit.nanometer)
        new_positions = unit.Quantity(np.zeros([n_atoms_new, 3]),
                                      unit=unit.nanometer)
        for idx in range(n_atoms_new):
            hyb_idx = self._new_to_hybrid_map[idx]
            new_positions[idx, :] = hybrid_positions[hyb_idx, :]
        return new_positions

    @property
    def hybrid_system(self):
        """
        The hybrid system.

        Returns
        -------
        hybrid_system : openmm.System
            The system representing a hybrid between old and new topologies
        """
        return self._hybrid_system

    @property
    def new_to_hybrid_atom_map(self):
        """
        Give a dictionary that maps new system atoms to the hybrid system.

        Returns
        -------
        new_to_hybrid_atom_map : dict of {int, int}
            The mapping of atoms from the new system to the hybrid
        """
        return self._new_to_hybrid_map

    @property
    def old_to_hybrid_atom_map(self):
        """
        Give a dictionary that maps old system atoms to the hybrid system.

        Returns
        -------
        old_to_hybrid_atom_map : dict of {int, int}
            The mapping of atoms from the old system to the hybrid
        """
        return self._old_to_hybrid_map

    @property
    def hybrid_positions(self):
        """
        The positions of the hybrid system. Dimensionality is (n_environment +
        n_core + n_old_unique + n_new_unique).
        The positions are assigned by first copying all the mapped positions
        from the old system in, then copying the mapped positions from the new
        system.

        Returns
        -------
        hybrid_positions : [n, 3] Quantity nanometers
        """
        return self._hybrid_positions

    @property
    def hybrid_topology(self):
        """
        An MDTraj hybrid topology for the purpose of writing out trajectories.
        
        Note that we do not expect this to be able to be parameterized by the
        openmm forcefield class.

        Returns
        -------
        hybrid_topology : mdtraj.Topology
        """
        return self._hybrid_topology

    @property
    def omm_hybrid_topology(self):
        """
        An OpenMM format of the hybrid topology. Also cannot be used to
        parameterize system, only to write out trajectories.

        Returns
        -------
        hybrid_topology : simtk.openmm.app.Topology
        """
        return mdt.Topology.to_openmm(self._hybrid_topology)
