# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Acknowledgements
----------------
Many parts of this restraints code comes from Yank
(https://github.com/choderalab/yank)
"""
import abc
import functools
import inspect
import numpy as np
import logging
import openmm
from openmm import unit as ommunit
import openmmtools as mmtools
from openmmtools import GlobalParameterState
import scipy


logger = logging.getLogger(__name__)


def methoddispatch(func):
    dispatcher = functools.singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    functools.update_wrapper(wrapper, dispatcher)

    return wrapper


class RestraintState(GlobalParameterState):
    """
    The state of a restraint.

    A ``ComposableState`` controlling the strength of a restraint
    through its ``lambda_restraints`` property.

    Parameters
    ----------
    parameters_name_suffix : str, optional
        If specified, the state will control the parameter
        ``lambda_restraint_[parameters_name_suffix]`` instead of just
        ``lambda_restraint``. This is useful if it's necessary to control
        multiple restraints.
    lambda_restraints : float
        The strength of the restraint. Must be between 0 and 1.

    Attributes
    ----------
    lambda_restraints

    Acknowledgement
    ---------------
    This originates from Yank (https://github.com/choderalab/yank).
    Please cite the following when using this code:
    https://zenodo.org/records/3534289

    Examples
    --------
    Create a system in a thermodynamic state.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(
    ...    system, 300*unit.kelvin
    ... )
    >>> sampler_state = states.SamplerState(positions)

    Apply a Harmonic restraint between receptor and protein. Let the restraint
    automatically determine all the parameters.

    >>> restraint = Harmonic()
    >>> restraint.determine_missing_parameters(
    ...    thermodynamic_state, sampler_state
    ... )
    >>> restraint.restrain_state(thermodynamic_state)

    Create a ``RestraintState`` object to control the strength of the
    restraint.

    >>> restraint_state = RestraintState(lambda_restraints=1.0)

    ``RestraintState`` implements the ``IComposableState`` interface, so it
    can be used with ``CompoundThermodynamicState``.

    >>> compound_state = states.CompoundThermodynamicState(
    ... thermodynamic_state=thermodynamic_state,
    ... composable_states=[restraint_state]
    ... )
    >>> compound_state.lambda_restraints
    1.0
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = compound_state.create_context(integrator)
    >>> context.getParameter('lambda_restraints')
    1.0

    You can control the parameters in the OpenMM Context by setting the state's
    attributes. To To deactivate the restraint, set `lambda_restraints` to 0.0.

    >>> compound_state.lambda_restraints = 0.0
    >>> compound_state.apply_to_context(context)
    >>> context.getParameter('lambda_restraints')
    0.0

    """
    lambda_restraints = GlobalParameterState.GlobalParameter(
        'lambda_restraints', standard_value=1.0
    )

    @lambda_restraints.validator
    def lambda_restraints(self, instance, new_value):
        if new_value is None:
            return None
        if not (0.0 <= new_value <= 1.0):
            raise ValueError('lambda_restraints must be between 0.0 and 1.0')
        return float(new_value)

    def __setstate__(self, serialization):
        super().__setstate__(serialization)


class ReceptorLigandRestraint(abc.ABC):
    """
    A restraint preventing a ligand from drifting too far from its receptor.

    With replica exchange simulations, keeping the ligand close to the binding
    pocket can enhance mixing between the interacting and the decoupled state.
    This should be always used in implicit simulation, where there are no
    periodic boundary conditions.

    This restraint strength is controlled by a global context parameter called
    ``lambda_restraints``. You can easily control this variable through the
    ``RestraintState`` object.

    Notes
    -----
    Creating a subclass requires the following:

        1. Implement a constructor. Optionally this can leave all or a subset
        of the restraint parameters undefined. In this case, you need to
        provide an implementation of :func:`determine_missing_parameters`.

        2. Implement :func:`restrain_state` that add the restrain ``Force`` to
        the state's`System`.

        3. Implement :func:`get_standard_state_correction` to return standard
        state correction.

        4. Optionally, implement :func:`determine_missing_parameters` to fill
        in the parameters left undefined in the constructor.

    """

    @abc.abstractmethod
    def restrain_state(self, thermodynamic_state):
        """Add the restraint force to the state's `System`.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.

        """
        pass

    @abc.abstractmethod
    def get_standard_state_correction(self, thermodynamic_state):
        """Return the standard state correction.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        """
        pass

    def determine_missing_parameters(
        self, thermodynamic_state, sampler_state, topology,
        ligand_smc, comp_resids
    ):
        """
        Automatically choose undefined parameters.

        Optionally, a :class:`ReceptorLigandRestraint` can support the
        automatic determination of all or a subset of the parameters that
        can be left undefined in the constructor, making implementation of
        this method optional.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynmaic state to inspect
        sampler_state : openmmtools.states.SamplerState
            The sampler state holding the positions of all atoms.
        topology : openmm.app.Topology
        ligand_smc : ligand SmallMoleculeComponent
        comp_resids : dict[Component, npt.NDArray]
        """
        errmsg = (f"{self.__class__.__name__} does not support automatic "
                  "determination of the restraint parameters")
        raise NotImplementedError(errmsg)

    @classmethod
    def _add_force_in_separate_group(cls, system, restraint_force):
        """
        Add the force to the System in a separate force group when possible.
        """
        # OpenMM supports a maximum of 32 force groups.
        available_force_groups = set(range(32))
        for force in system.getForces():
            available_force_groups.discard(force.getForceGroup())

        # If the System is full, just separate the force from nonbonded interactions.
        if len(available_force_groups) == 0:
            _, nonbonded_force = mmtools.forces.find_forces(
                system,
                openmm.NonbondedForce,
                only_one=True
            )
            available_force_groups = set(range(32))
            available_force_groups.discard(nonbonded_force.getForceGroup())

        restraint_force.setForceGroup(min(available_force_groups))
        system.addForce(restraint_force)

    @property
    def _excluded_init_parameters(self):
        """
        List of excluded parameters from the :func:`__init__` call to ensure
        all non-atom selection parameters are defined
        """
        return ['self', 'restrained_receptor_atoms', 'restrained_ligand_atoms']

    @property
    def _parameters(self):
        """dict: restraint parameters in dict forms."""
        argspec = inspect.getfullargspec(self.__init__)
        parameter_names = argspec.args

        # Exclude non-parameters arguments.
        for exclusion in self._excluded_init_parameters:
            parameter_names.remove(exclusion)

        # Retrieve and store options.
        parameters = {parameter_name: getattr(self, parameter_name)
                      for parameter_name in parameter_names}
        return parameters


class _RestrainedAtomsProperty(object):
    """
    Descriptor of restrained atoms.

    Casts generic iterables of ints into lists.
    """

    def __init__(self, atoms_type):
        self._atoms_type = atoms_type

    @property
    def _attribute_name(self):
        """Name of the internally stored variable (read-only)."""
        return f"_restrained_{self._atoms_type}_atoms"

    def __get__(self, instance, owner_class=None):
        return getattr(instance, self._attribute_name)

    def __set__(self, instance, new_restrained_atoms):
        # If we set the restrained attributes to None
        # no reason to check things.
        if new_restrained_atoms is not None:
            new_restrained_atoms = self._validate_atoms(new_restrained_atoms)
        setattr(instance, self._attribute_name, new_restrained_atoms)

    @methoddispatch
    def _validate_atoms(self, restrained_atoms):
        """
        Casts a generic iterable of ints into a list to support
        concatenation.
        """
        try:
            restrained_atoms = restrained_atoms.tolist()
        except AttributeError:
            restrained_atoms = list(restrained_atoms)
        return restrained_atoms


class BoreschLike(ReceptorLigandRestraint, abc.ABC):
    """
    Abstract class to impose Boresch-like orientational restraints on
    protein-ligand system. Subclasses are specific implementations
    of the energy functions

    This restraints the ligand binding mode by constraining 1 distance, 2
    angles and 3 dihedrals between 3 atoms of the receptor and 3 atoms of
    the ligand.

    More precisely, the energy expression of the restraint is given by

        .. code-block:: python

            E = lambda_restraints * {
                    K_r/2 * [|r3 - l1| - r_aA0]^2 +
                    + K_thetaA/2 * [angle(r2,r3,l1) - theta_A0]^2 +
                    + K_thetaB/2 * [angle(r3,l1,l2) - theta_B0]^2 +
                    + K_phiA/2 * hav(dihedral(r1,r2,r3,l1) - phi_A0) * 2 +
                    + K_phiB/2 * hav(dihedral(r2,r3,l1,l2) - phi_B0) * 2 +
                    + K_phiC/2 * hav(dihedral(r3,l1,l2,l3) - phi_C0) * 2
                }

    , where ``hav`` is the Haversine function ``(1-cos(x))/2`` and the
    parameters are:

        ``r1``, ``r2``, ``r3``: the coordinates of the 3 receptor atoms.

        ``l1``, ``l2``, ``l3``: the coordinates of the 3 ligand atoms.

        ``K_r``: the spring constant for the restrained distance ``|r3 - l1|``.

        ``r_aA0``: the equilibrium distance of ``|r3 - l1|``.

        ``K_thetaA``, ``K_thetaB``: the spring constants for ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        ``theta_A0``, ``theta_B0``: the equilibrium angles of ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        ``K_phiA``, ``K_phiB``, ``K_phiC``: the spring constants for ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        ``phi_A0``, ``phi_B0``, ``phi_C0``: the equilibrium torsion of ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        ``lambda_restraints``: a scale factor that can be used to control the
        strength of the restraint.

    You can control ``lambda_restraints`` through the class
    :class:`RestraintState`.

    The class supports automatic determination of the parameters left undefined
    in the constructor through :func:`determine_missing_parameters`.


    This function used to be based on the Boresch orientational restraints [1]
    and has similar form to its energy equation

        .. code-block:: python

            E = lambda_restraints * {
                    K_r/2 * [|r3 - l1| - r_aA0]^2 +
                    + K_thetaA/2 * [angle(r2,r3,l1) - theta_A0]^2 +
                    + K_thetaB/2 * [angle(r3,l1,l2) - theta_B0]^2 +
                    + K_phiA/2 * [dihedral(r1,r2,r3,l1) - phi_A0]^2 +
                    + K_phiB/2 * [dihedral(r2,r3,l1,l2) - phi_B0]^2 +
                    + K_phiC/2 * [dihedral(r3,l1,l2,l3) - phi_C0]^2
                }

    However, the form at the top is periodic with the dihedral angle and
    imposes a more steep energy penalty while still maintaining the same
    Taylor series expanded force and energy near phi_X0. The ``*2`` on the
    ``hav()`` functions in the energy expression are shown as the explicit
    correction to the ``hav()`` function to make the leading spring constant
    force consistent with the original harmonic Boresch restraint. In practice,
    the ``1/2`` from the ``hav()`` function is omitted.


    *Warning*: Symmetry corrections for symmetric ligands are not
    automatically applied. See Ref [1] and [2] for more information
    on correcting for ligand symmetry.

    *Warning*: Only heavy atoms can be restrained. Hydrogens will
    automatically be excluded.

    Parameters
    ----------
    restrained_receptor_atoms : iterable of int, str, or None; Optional
        The indices of the receptor atoms to restrain, an MDTraj DSL
        expression, or a :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        If this is a list of three ints, the receptor atoms will be
        restrained in order, r1, r2, r3. If there are more than three
        entries or the selection string resolves more than three atoms,
        the three restrained atoms will be chosen at random from
        the selection. This can temporarily be left undefined,
        but ``determine_missing_parameters()`` must be called before
        using the Restraint object. The same if a DSL expression or
        Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, str, or None; Optional
        The indices of the ligand atoms to restrain, an MDTraj DSL
        expression, or a :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        If this is a list of three ints, the receptor atoms will be restrained
        in order, l1, l2, l3. If there are more than three entries or the
        selection string resolves more than three atoms, the three restrained
        atoms will be chosen at random from the selection.
        This can temporarily be left undefined, but
        ``determine_missing_parameters()`` must be called before using the
        Restraint object. The same if a DSL expression or Topography region
        is provided (default is None).
    K_r : simtk.unit.Quantity, optional
        The spring constant for the restrained distance ``|r3 - l1|`` (units
        compatible with kilocalories_per_mole/angstrom**2).
    r_aA0 : simtk.unit.Quantity, optional
        The equilibrium distance between r3 and l1 (units of length).
    K_thetaA, K_thetaB : simtk.unit.Quantity, optional
        The spring constants for ``angle(r2, r3, l1)`` and 
        `angle(r3, l1, l2)`` (units compatible with
        kilocalories_per_mole/radians**2).
    theta_A0, theta_B0 : simtk.unit.Quantity, optional
        The equilibrium angles of ``angle(r2, r3, l1)`` and
        ``angle(r3, l1, l2)`` (units compatible with radians).
    K_phiA, K_phiB, K_phiC : simtk.unit.Quantity, optional
        The spring constants for ``dihedral(r1, r2, r3, l1)``,
        ``dihedral(r2, r3, l1, l2)`` and ``dihedral(r3,l1,l2,l3)``
        (units compatible with kilocalories_per_mole/radians**2).
    phi_A0, phi_B0, phi_C0 : simtk.unit.Quantity, optional
        The equilibrium torsion of ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)`` and ``dihedral(r3,l1,l2,l3)``
        (units compatible with radians).

    Attributes
    ----------
    restrained_receptor_atoms : list of int
        The indices of the 3 receptor atoms to restrain [r1, r2, r3].
    restrained_ligand_atoms : list of int
        The indices of the 3 ligand atoms to restrain [l1, l2, l3].

    References
    ----------
    [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839
    [2] Mobley DL, Chodera JD, and Dill KA. J Chem Phys 125:084902, 2006.
        https://dx.doi.org/10.1063%2F1.2221683

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 298*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor
    atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    Create a partially defined restraint

    >>> restraint = Boresch(restrained_receptor_atoms=[1335, 1339, 1397],
    ...                     restrained_ligand_atoms=[2609, 2607, 2606],
    ...                     K_r=20.0*unit.kilocalories_per_mole/unit.angstrom**2,
    ...                     r_aA0=0.35*unit.nanometer)

    and automatically identify the other parameters. When trying to impose
    a restraint with undefined parameters, RestraintParameterError is raised.

    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)

    """
    def __init__(self, restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_r=None, r_aA0=None,
                 K_thetaA=None, theta_A0=None,
                 K_thetaB=None, theta_B0=None,
                 K_phiA=None, phi_A0=None,
                 K_phiB=None, phi_B0=None,
                 K_phiC=None, phi_C0=None):
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.restrained_ligand_atoms = restrained_ligand_atoms
        self.K_r = K_r
        self.r_aA0 = r_aA0
        self.K_thetaA, self.K_thetaB = K_thetaA, K_thetaB
        self.theta_A0, self.theta_B0 = theta_A0, theta_B0
        self.K_phiA, self.K_phiB, self.K_phiC = K_phiA, K_phiB, K_phiC
        self.phi_A0, self.phi_B0, self.phi_C0 = phi_A0, phi_B0, phi_C0

    # -------------------------------------------------------------------------
    # Public properties.
    # -------------------------------------------------------------------------

    class _BoreschRestrainedAtomsProperty(_RestrainedAtomsProperty):
        """
        Descriptor of restrained atoms.

        Extends `_RestrainedAtomsProperty` to handle single integers and strings.
        """

        _DEBUG_MSG = ('You are specifying {} {} atoms, '
                      'the final atoms will be chosen at from this set '
                      'after calling "determine_missing_parameters()"')

        @methoddispatch
        def _validate_atoms(self, restrained_atoms):
            restrained_atoms = super()._validate_atoms(restrained_atoms)
            if len(restrained_atoms) < 3:
                raise ValueError('At least three {} atoms are required to impose a '
                                 'Boresch-style restraint.'.format(self._atoms_type))
            elif len(restrained_atoms) > 3:
                logger.debug(self._DEBUG_MSG.format("more than three", self._atoms_type))
            return restrained_atoms

        @_validate_atoms.register(str)
        def _cast_atom_string(self, restrained_atoms):
            logger.debug(self._DEBUG_MSG.format("a string for", self._atoms_type))
            return restrained_atoms

    restrained_receptor_atoms = _BoreschRestrainedAtomsProperty('receptor')
    restrained_ligand_atoms = _BoreschRestrainedAtomsProperty('ligand')

    # -------------------------------------------------------------------------
    # Public methods.
    # -------------------------------------------------------------------------

    def restrain_state(self, thermodynamic_state):
        """Add the restraint force to the state's ``System``.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.

        """
        # TODO replace dihedral restraints with negative log von Mises distribution?
        #       https://en.wikipedia.org/wiki/Von_Mises_distribution, the von Mises parameter
        #       kappa would be computed from the desired standard deviation (kappa ~ sigma**(-2))
        #       and the standard state correction would need to be modified.

        # Check if all parameters are defined.
        self._check_parameters_defined()

        energy_function = self._get_energy_function_string()

        # Add constant definitions to the energy function
        for name, value in self._parameters.items():
            energy_function += '%s = %f; ' % (name, value.value_in_unit_system(ommunit.md_unit_system))

        # Create the force
        n_particles = 6  # number of particles involved in restraint: p1 ... p6
        restraint_force = openmm.CustomCompoundBondForce(n_particles, energy_function)
        restraint_force.addGlobalParameter('lambda_restraints', 1.0)
        restraint_force.addBond(self.restrained_receptor_atoms + self.restrained_ligand_atoms, [])
        restraint_force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        self._add_force_in_separate_group(system, restraint_force)
        thermodynamic_state.system = system

    def get_standard_state_correction(self, thermodynamic_state):
        """Return the standard state correction.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.

        Returns
        -------
        DeltaG : float
           Computed standard-state correction in dimensionless units (kT).

        """
        self._check_parameters_defined()

        def strip(passed_unit):
            """Cast the passed_unit into md unit system for integrand lambda functions"""
            return passed_unit.value_in_unit_system(ommunit.md_unit_system)

        # Shortcuts variables.
        pi = np.pi
        kT = thermodynamic_state.kT
        p = self  # For the parameters.
        V0 = 1660.53928 * ommunit.angstroms**3 

        # Radial
        sigma = 1 / ommunit.sqrt(p.K_r / kT)
        rmin = strip(max(0 * ommunit.angstrom, p.r_aA0 - 8 * sigma))
        rmax = strip(p.r_aA0 + 8 * sigma)
        r0 = strip(p.r_aA0)
        K_r = strip(p.K_r)
        I = lambda r: self._numerical_distance_integrand(r, r0, K_r, strip(kT))
        integral_packet = scipy.integrate.quad(I, rmin, rmax) * ommunit.nanometer**3
        ExpDeltaG = integral_packet[0]

        # Angular
        for name in ['A', 'B']:
            theta0 = strip(getattr(p, 'theta_' + name + '0'))
            K_theta = strip(getattr(p, 'K_theta' + name))
            I = lambda theta: self._numerical_angle_integrand(
                theta, theta0, K_theta, strip(kT)
            )
            integral_packet = scipy.integrate.quad(I, 0, pi)
            ExpDeltaG *= integral_packet[0]

        # Torsion
        for name in ['A', 'B', 'C']:
            phi0 = strip(getattr(p, 'phi_' + name + '0'))
            K_phi = strip(getattr(p, 'K_phi' + name))
            I = lambda phi: self._numerical_torsion_integrand(
                phi, phi0, K_phi, strip(kT)
            )
            integral_packet = scipy.integrate.quad(I, -pi, pi)
            ExpDeltaG *= integral_packet[0]

        DeltaG = -np.log(8 * pi**2 * V0 / ExpDeltaG)
        return DeltaG

    def determine_missing_parameters(
        self, thermodynamic_state, sampler_state, topology, ligand_smc,
        comp_resids
    ):
        """Determine parameters and restrained atoms automatically.

        Currently, all equilibrium values are measured from the initial structure,
        while spring constants are set to 20 kcal/(mol A**2) or 20 kcal/(mol rad**2)
        as in Ref [1]. The restrained atoms are selected so that the analytical
        standard state correction will be valid.

        Parameters that have been already specified are left untouched.

        Future iterations of this feature will introduce the ability to extract
        equilibrium parameters and spring constants from a short simulation.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state.
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topology: openmm.app.Topology
        ligand_smc : ligand SmallMoleculeComponent
        comp_resids : dict[Component, npt.NDArray]
        """

        logger.debug('Automatically selecting restraint atoms and parameters:')

        # If restrained atoms are already specified, we only need to determine parameters.
        restrained_atoms = self._pick_restrained_atoms(
            sampler_state, topology, ligand_smc, comp_resids
        )
        self.restrained_receptor_atoms = restrained_atoms[:3]
        self.restrained_ligand_atoms = restrained_atoms[3:]

        # Determine restraint parameters for these atoms. -- maybe need to add stuff here
        self._determine_restraint_parameters(sampler_state, topology)

        ssc = self.get_standard_state_correction(thermodynamic_state)
        logger.debug('Standard state correction: {} kT'.format(ssc))

    # -------------------------------------------------------------------------
    # Abstract Functions
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def _get_energy_function_string(self):
        """
        Get the energy function string which defines the full restraint
        compatible with OpenMM Custom*Force expressions.

        Restraint constants can be used in this function and will be
        substituted in

          * K_r, r_aA0
          * K_thetaA, theta_A0,
          * K_thetaB, theta_B0,
          * K_phiA, phi_A0,
          * K_phiB, phi_B0,
          * K_phiC, phi_C0

        Variables should also be used:

          * lambda_restraints    : Alchemical variable, should be scalar on whole energy
          * distance(p3,p4)      : Restrained distance
          * angle(p2,p3,p4)      : Restrained angle "A"
          * angle(p3,p4,p5)      : Restrained angle "B"
          * dihedral(p1,p2,p3,p4): Restrained torsion "A"
          * dihedral(p2,p3,p4,p5): Restrained torsion "B"
          * dihedral(p3,p4,p5,p6): Restrained torsion "C"

        Returns
        -------
        energy_function : string
            String defining the force compatible with OpenMM Custom
        """
        pass

    @abc.abstractmethod
    def _numerical_torsion_integrand(self, phi, phi0, spring_constant, kt):
        """
        Integrand for the torsion (phi) restraints which will be integrated
        numerically for standard state correction

        Domain is on [-pi, pi], the same domain OpenMM uses

        Parameters
        ----------
        phi : float or np.ndarray of float
            Torsion angle which will be integrated, units of radians
        phi0 : float
            Equilibrium torsion angle at which force of restraint often is 0,
            units of radians
        spring_constant : float
            Spring constant for this torsion in units of kJ/mol/nm**2
        kt : float
            Boltzmann Temperature of the thermodynamic state restraining the
            atoms = kB * T in units of kJ/mol

        Returns
        -------
        integrand : float
            Value of the integrated
        """
        pass

    @abc.abstractmethod
    def _numerical_angle_integrand(self, theta, theta0, spring_constant, kt):
        """
        Integrand for the angle (theta) restraints which will be integrated
        numerically for standard state correction

        Domain is on [0, pi]

        Parameters
        ----------
        theta : float or np.ndarray of float
            Angle which will be integrated, units of radians
        theta0 : float
            Equilibrium angle at which force of restraint often is 0, units of
            radians
        spring_constant : float
            Spring constant for this angle in units of with kJ/mol/nm**2
        kt : float
            Boltzmann Temperature of the thermodynamic state restraining the
            atoms = kB * T in units of kJ/mol

        Returns
        -------
        integrand : float
            Value of the integrated
        """
        pass

    @abc.abstractmethod
    def _numerical_distance_integrand(self, r, r0, spring_constant, kt):
        """
        Integrand for the distance restraint which will be integrated
        numerically for standard state correction

        Domain is on [0, +infinity]

        Parameters
        ----------
        r : float or np.ndarray of float
           Distance which will be integrated, units of nm
        r0 : float
            Equilibrium distance at which force of restraint often is 0, nm
        spring_constant : float
            Spring constant for this distance in units of kJ/mol/nm**2
        kt : float
            Boltzmann Temperature of the thermodynamic state restraining the
            atoms = kB * T in units of kJ/mol

        Returns
        -------
        integrand : float
            Value of the integrated
        """
        pass

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _check_parameters_defined(self):
        """Raise an exception there are still parameters undefined."""
        if not self._are_restrained_atoms_defined:
            raise ValueError('Undefined restrained atoms.')

        # Find undefined parameters and raise error.
        undefined_parameters = [name for name, value in self._parameters.items() if value is None]
        if len(undefined_parameters) > 0:
            err_msg = ('Undefined parameters for Boresch-like restraint:'
                       "{undefined_parameters}'")
            raise ValueError(err_msg)

    @property
    def _are_restrained_atoms_defined(self):
        """
        Check if the restrained atoms are defined well enough to make a
        restraint.
        """
        for atoms in [self.restrained_receptor_atoms,
                      self.restrained_ligand_atoms]:
            # Atoms should be a list or None at this point due to the
            # _RestrainedAtomsProperty class
            if atoms is None or not (isinstance(atoms, list) and len(atoms) == 3):
                return False
        return True

    @staticmethod
    def _is_collinear(positions, atoms, threshold=0.9):
        """Report whether any sequential vectors in a sequence of atoms are collinear.

        Parameters
        ----------
        positions : n_atoms x 3 simtk.unit.Quantity
            Reference positions to use for imposing restraints (units of length).
        atoms : iterable of int
            The indices of the atoms to test.
        threshold : float, optional, default=0.9
            Atoms are not collinear if their sequential vector separation dot
            products are less than ``threshold``.

        Returns
        -------
        result : bool
            Returns True if any sequential pair of vectors is collinear; False otherwise.

        """
        result = False
        for i in range(len(atoms)-2):
            v1 = positions[atoms[i+1], :] - positions[atoms[i], :]
            v2 = positions[atoms[i+2], :] - positions[atoms[i+1], :]
            normalized_inner_product = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
            result = result or (normalized_inner_product > threshold)

        return result

    def _pick_restrained_atoms(
        self, sampler_state, topology, ligand_smc, comp_resids
    ):
        """Select atoms to be used in restraint.

        Parameters
        ----------
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topography : yank.Topography, optional
            The topography with labeled receptor and ligand atoms.

        Returns
        -------
        restrained_atoms : list of int
            List of six atom indices used in the restraint.
            restrained_atoms[0:3] belong to the receptor,
            restrained_atoms[4:6] belong to the ligand.

        Notes
        -----
        The current algorithm simply selects random subsets of receptor
        and ligand atoms and rejects those that are too close to collinear.
        Future updates can further refine this algorithm.

        """
        # If receptor and ligand atoms are explicitly provided, use those.
        heavy_ligand_atoms = self.restrained_ligand_atoms
        heavy_receptor_atoms = self.restrained_receptor_atoms

        # Otherwise we restrain only heavy atoms.
        heavy_atoms = set(topography.topology.select('not element H').tolist())
        # Intersect heavy atoms with receptor/ligand atoms (s1&s2 is intersect).

        atom_selector = _AtomSelector(topography)

        heavy_ligand_atoms = atom_selector.compute_atom_intersect(heavy_ligand_atoms, 'ligand_atoms', heavy_atoms)
        heavy_receptor_atoms = atom_selector.compute_atom_intersect(heavy_receptor_atoms, 'receptor_atoms', heavy_atoms)

        if len(heavy_receptor_atoms) < 3 or len(heavy_ligand_atoms) < 3:
            raise ValueError('There must be at least three heavy atoms in receptor_atoms '
                             '(# heavy {}) and ligand_atoms (# heavy {}).'.format(
                                     len(heavy_receptor_atoms), len(heavy_ligand_atoms)))

        # If r3 or l1 atoms are given. We have to pick those.
        if isinstance(heavy_receptor_atoms, list):
            r3_atoms = [heavy_receptor_atoms[2]]
        else:
            r3_atoms = heavy_receptor_atoms
        if isinstance(heavy_ligand_atoms, list):
            l1_atoms = [heavy_ligand_atoms[0]]
        else:
            l1_atoms = heavy_ligand_atoms
        # TODO: Cast itertools generator to np array more efficiently
        r3_l1_pairs = np.array(list(itertools.product(r3_atoms, l1_atoms)))

        # Filter r3-l1 pairs that are too close/far away for the distance constraint.
        max_distance = 4 * unit.angstrom/unit.nanometer
        min_distance = 1 * unit.angstrom/unit.nanometer
        t = md.Trajectory(sampler_state.positions / unit.nanometers, topography.topology)
        distances = md.geometry.compute_distances(t, r3_l1_pairs)[0]
        indices_of_in_range_pairs = np.where(np.logical_and(distances > min_distance, distances <= max_distance))[0]

        if len(indices_of_in_range_pairs) == 0:
            error_msg = ('There are no heavy ligand atoms within the range of [{},{}] nm heavy receptor atoms!\n'
                         'Please Check your input files or try another restraint class')
            raise ValueError(error_msg.format(min_distance, max_distance))
        r3_l1_pairs = r3_l1_pairs[indices_of_in_range_pairs].tolist()

        def find_bonded_to(input_atom_index, comparison_set):
            """
            Find bonded network between the atoms to create a selection with 1 angle to the reference

            Parameters
            ----------
            input_atom_index : int
                Reference atom index to try and create the selection from the bonds
            comparison_set : iterable of int
                Set of additional atoms to try and make the selection from. There should be at least
                one non-colinear set 3 atoms which are bonded together in R-B-C where R is the input_atom_index
                and B, C are atoms in the comparison_set bonded to each other.
                Can be inclusive of input_atom_index and C can be bound to R as well as B

            Returns
            -------
            bonded_atoms : list of int, length 3
                Returns the list of atoms in order of input_atom_index <- bonded atom <- bonded atom
            """
            # Probably could make this faster if we added a graph module like networkx dep, but not needed
            # Could also be done by iterating over OpenMM System angles
            # Get topology
            top = topography.topology
            bonds = np.zeros([top.n_atoms, top.n_atoms], dtype=bool)
            # Create bond graph
            for a1, a2 in top.bonds:
                a1 = a1.index
                a2 = a2.index
                bonds[a1, a2] = bonds[a2, a1] = True
            all_bond_options = []
            # Cycle through all bonds on the reference
            for a2, first_bond in enumerate(bonds[input_atom_index]):
                # Enumerate all secondary bonds from the reference but only if in comparison set
                if first_bond and a2 in comparison_set:
                    # Same as first
                    for a3, second_bond in enumerate(bonds[a2]):
                        if second_bond and a3 in comparison_set and a3 != input_atom_index:
                            all_bond_options.append([a2, a3])
            # This will raise a ValueError if nothing is found
            return random.sample(all_bond_options, 1)[0]

        # Iterate until we have found a set of non-collinear atoms.
        accepted = False
        max_attempts = 100
        attempts = 0
        while not accepted:
            logger.debug('Attempt {} / {} at automatically selecting atoms and '
                         'restraint parameters...'.format(attempts, max_attempts))
            # Select a receptor/ligand atom in range of each other for the distance constraint.
            r3_l1_atoms = random.sample(r3_l1_pairs, 1)[0]
            # Determine remaining receptor/ligand atoms.
            if isinstance(heavy_receptor_atoms, list):
                r1_r2_atoms = heavy_receptor_atoms[:2]
            else:
                try:
                    r1_r2_atoms = find_bonded_to(r3_l1_atoms[0], heavy_receptor_atoms)[::-1]
                except ValueError:
                    r1_r2_atoms = None
            if isinstance(heavy_ligand_atoms, list):
                l2_l3_atoms = heavy_ligand_atoms[1:]
            else:
                try:
                    l2_l3_atoms = find_bonded_to(r3_l1_atoms[-1], heavy_ligand_atoms)
                except ValueError:
                    l2_l3_atoms = None
            # Reject collinear sets of atoms.
            if r1_r2_atoms is None or l2_l3_atoms is None:
                accepted = False
            else:
                restrained_atoms = r1_r2_atoms + r3_l1_atoms + l2_l3_atoms
                accepted = not self._is_collinear(sampler_state.positions, restrained_atoms)
            if attempts > max_attempts:
                raise RuntimeError("Could not find any good sets of bonded atoms to make stable Boresch-like "
                                   "restraints from. There should be at least 1 real defined angle in the"
                                   "selected restrained ligand atoms and 1 in the selected restrained receptor atoms "
                                   "for good numerical stability")
            else:
                attempts += 1

        logger.debug('Selected atoms to restrain: {}'.format(restrained_atoms))
        return restrained_atoms

    def _determine_restraint_parameters(self, sampler_states, topography):
        """Determine restraint parameters.

        Currently, all equilibrium values are measured from the initial structure,
        while spring constants are set to 20 kcal/(mol A**2) or 20 kcal/(mol rad**2)
        as in [1].

        Future iterations of this feature will introduce the ability to extract
        equilibrium parameters and spring constants from a short simulation.

        References
        ----------
        [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839

        """
        # We determine automatically only the parameters that have been left undefined.
        def _assign_if_undefined(attr_name, attr_value):
            """Assign value to self.name only if it is None."""
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, attr_value)

        # Merge receptor and ligand atoms in a single array for easy manipulation.
        restrained_atoms = self.restrained_receptor_atoms + self.restrained_ligand_atoms

        # Set spring constants uniformly, as in Ref [1] Table 1 caption.
        _assign_if_undefined('K_r', 20.0 * unit.kilocalories_per_mole / unit.angstrom**2)
        for parameter_name in ['K_thetaA', 'K_thetaB', 'K_phiA', 'K_phiB', 'K_phiC']:
            _assign_if_undefined(parameter_name, 20.0 * unit.kilocalories_per_mole / unit.radian**2)

        # Measure equilibrium geometries from static reference structure
        t = md.Trajectory(sampler_states.positions / unit.nanometers, topography.topology)

        atom_pairs = [restrained_atoms[2:4]]
        distances = md.geometry.compute_distances(t, atom_pairs, periodic=False)
        _assign_if_undefined('r_aA0', distances[0][0] * unit.nanometers)

        atom_triplets = [restrained_atoms[i:(i+3)] for i in range(1, 3)]
        angles = md.geometry.compute_angles(t, atom_triplets, periodic=False)
        for parameter_name, angle in zip(['theta_A0', 'theta_B0'], angles[0]):
            _assign_if_undefined(parameter_name, angle * unit.radians)

        atom_quadruplets = [restrained_atoms[i:(i+4)] for i in range(3)]
        dihedrals = md.geometry.compute_dihedrals(t, atom_quadruplets, periodic=False)
        for parameter_name, angle in zip(['phi_A0', 'phi_B0', 'phi_C0'], dihedrals[0]):
            _assign_if_undefined(parameter_name, angle * unit.radians)

        # Write restraint parameters
        msg = 'restraint parameters:\n'
        for parameter_name, parameter_value in self._parameters.items():
            msg += '%24s : %s\n' % (parameter_name, parameter_value)
        logger.debug(msg)


class Boresch(BoreschLike):
    """
    Impose Boresch-style orientational restraints on protein-ligand system.

    This restraints the ligand binding mode by constraining 1 distance, 2
    angles and 3 dihedrals between 3 atoms of the receptor and 3 atoms of
    the ligand.

    More precisely, the energy expression of the restraint is given by

        .. code-block:: python

            E = lambda_restraints * {
                    K_r/2 * [|r3 - l1| - r_aA0]^2 +
                    + K_thetaA/2 * [angle(r2,r3,l1) - theta_A0]^2 +
                    + K_thetaB/2 * [angle(r3,l1,l2) - theta_B0]^2 +
                    + K_phiA/2 * [dihedral(r1,r2,r3,l1) - phi_A0]^2 +
                    + K_phiB/2 * [dihedral(r2,r3,l1,l2) - phi_B0]^2 +
                    + K_phiC/2 * [dihedral(r3,l1,l2,l3) - phi_C0]^2
                }

    , where the parameters are:

        ``r1``, ``r2``, ``r3``: the coordinates of the 3 receptor atoms.

        ``l1``, ``l2``, ``l3``: the coordinates of the 3 ligand atoms.

        ``K_r``: the spring constant for the restrained distance ``|r3 - l1|``.

        ``r_aA0``: the equilibrium distance of ``|r3 - l1|``.

        ``K_thetaA``, ``K_thetaB``: the spring constants for ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        ``theta_A0``, ``theta_B0``: the equilibrium angles of ``angle(r2,r3,l1)`` and ``angle(r3,l1,l2)``.

        ``K_phiA``, ``K_phiB``, ``K_phiC``: the spring constants for ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        ``phi_A0``, ``phi_B0``, ``phi_C0``: the equilibrium torsion of ``dihedral(r1,r2,r3,l1)``,
        ``dihedral(r2,r3,l1,l2)``, ``dihedral(r3,l1,l2,l3)``.

        ``lambda_restraints``: a scale factor that can be used to control the strength
        of the restraint.

    You can control ``lambda_restraints`` through the class :class:`RestraintState`.

    The class supports automatic determination of the parameters left undefined
    in the constructor through :func:`determine_missing_parameters`.

    *Warning*: Symmetry corrections for symmetric ligands are not automatically applied.
    See Ref [1] and [2] for more information on correcting for ligand symmetry.

    *Warning*: Only heavy atoms can be restrained. Hydrogens will automatically be excluded.

    Parameters
    ----------
    restrained_receptor_atoms : iterable of int, str, or None; Optional
        The indices of the receptor atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        If this is a list of three ints, the receptor atoms will be restrained in order, r1, r2, r3. If there are more
        than three entries or the selection string resolves more than three atoms, the three restrained atoms will
        be chosen at random from the selection.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    restrained_ligand_atoms : iterable of int, str, or None; Optional
        The indices of the ligand atoms to restrain, an MDTraj DSL expression, or a
        :class:`Topography <yank.Topography>` region name,
        or :func:`Topography Select String <yank.Topography.select>`.
        If this is a list of three ints, the receptor atoms will be restrained in order, l1, l2, l3. If there are more
        than three entries or the selection string resolves more than three atoms, the three restrained atoms will
        be chosen at random from the selection.
        This can temporarily be left undefined, but ``determine_missing_parameters()``
        must be called before using the Restraint object. The same if a DSL
        expression or Topography region is provided (default is None).
    K_r : simtk.unit.Quantity, optional
        The spring constant for the restrained distance ``|r3 - l1|`` (units
        compatible with kilocalories_per_mole/angstrom**2).
    r_aA0 : simtk.unit.Quantity, optional
        The equilibrium distance between r3 and l1 (units of length).
    K_thetaA, K_thetaB : simtk.unit.Quantity, optional
        The spring constants for ``angle(r2, r3, l1)`` and ``angle(r3, l1, l2)``
        (units compatible with kilocalories_per_mole/radians**2).
    theta_A0, theta_B0 : simtk.unit.Quantity, optional
        The equilibrium angles of ``angle(r2, r3, l1)`` and ``angle(r3, l1, l2)``
        (units compatible with radians).
    K_phiA, K_phiB, K_phiC : simtk.unit.Quantity, optional
        The spring constants for ``dihedral(r1, r2, r3, l1)``,
        ``dihedral(r2, r3, l1, l2)`` and ``dihedral(r3,l1,l2,l3)`` (units compatible
        with kilocalories_per_mole/radians**2).
    phi_A0, phi_B0, phi_C0 : simtk.unit.Quantity, optional
        The equilibrium torsion of ``dihedral(r1,r2,r3,l1)``, ``dihedral(r2,r3,l1,l2)``
        and ``dihedral(r3,l1,l2,l3)`` (units compatible with radians).

    Attributes
    ----------
    restrained_receptor_atoms : list of int
        The indices of the 3 receptor atoms to restrain [r1, r2, r3].
    restrained_ligand_atoms : list of int
        The indices of the 3 ligand atoms to restrain [l1, l2, l3].

    References
    ----------
    [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
        http://dx.doi.org/10.1021/jp0217839
    [2] Mobley DL, Chodera JD, and Dill KA. J Chem Phys 125:084902, 2006.
        https://dx.doi.org/10.1063%2F1.2221683

    Examples
    --------
    Create the ThermodynamicState.

    >>> from openmmtools import testsystems, states
    >>> system_container = testsystems.LysozymeImplicit()
    >>> system, positions = system_container.system, system_container.positions
    >>> thermodynamic_state = states.ThermodynamicState(system, 298*unit.kelvin)
    >>> sampler_state = states.SamplerState(positions)

    Identify ligand atoms. Topography automatically identify receptor atoms too.

    >>> from yank.yank import Topography
    >>> topography = Topography(system_container.topology, ligand_atoms=range(2603, 2621))

    Create a partially defined restraint

    >>> restraint = Boresch(restrained_receptor_atoms=[1335, 1339, 1397],
    ...                     restrained_ligand_atoms=[2609, 2607, 2606],
    ...                     K_r=20.0*unit.kilocalories_per_mole/unit.angstrom**2,
    ...                     r_aA0=0.35*unit.nanometer)

    and automatically identify the other parameters. When trying to impose
    a restraint with undefined parameters, RestraintParameterError is raised.

    >>> try:
    ...     restraint.restrain_state(thermodynamic_state)
    ... except RestraintParameterError:
    ...     print('There are undefined parameters. Choosing restraint parameters automatically.')
    ...     restraint.determine_missing_parameters(thermodynamic_state, sampler_state, topography)
    ...     restraint.restrain_state(thermodynamic_state)
    ...
    There are undefined parameters. Choosing restraint parameters automatically.

    Get standard state correction.

    >>> correction = restraint.get_standard_state_correction(thermodynamic_state)
    """

    def _get_energy_function_string(self):
        """
        Get the energy function string which defines the full restraint compatible with OpenMM Custom*Force
        expressions.

        Returns
        -------
        energy_function : string
            String defining the force compatible with OpenMM Custom
        """
        energy_function = """
            lambda_restraints * E;
            E = (K_r/2)*(distance(p3,p4) - r_aA0)^2
            + (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2
            + (K_phiA/2)*dphi_A^2 + (K_phiB/2)*dphi_B^2 + (K_phiC/2)*dphi_C^2;
            dphi_A = dA - floor(dA/(2*pi)+0.5)*(2*pi); dA = dihedral(p1,p2,p3,p4) - phi_A0;
            dphi_B = dB - floor(dB/(2*pi)+0.5)*(2*pi); dB = dihedral(p2,p3,p4,p5) - phi_B0;
            dphi_C = dC - floor(dC/(2*pi)+0.5)*(2*pi); dC = dihedral(p3,p4,p5,p6) - phi_C0;
            pi = %f;
            """ % np.pi
        return energy_function

    def _numerical_distance_integrand(self, r, r0, spring_constant, kt):
        """
        Integrand for the distance restraint which will be integrated numerically for standard state correction

        Domain is on [0, +infinity], practically is only taken up to 8 thermal fluctuations.

        Parameters
        ----------
        r : float or np.ndarray of float
           Distance which will be integrated, units of nm
        r0 : float
            Equilibrium distance at which force of restraint often is 0, nm
        spring_constant : float
            Spring constant for this distance in units of kJ/mol/nm**2
        kt : float
            Boltzmann Temperature of the thermodynamic state restraining the atoms = kB * T
            in units of kJ/mol

        Returns
        -------
        integrand : float
            Value of the integrated
        """
        return r ** 2 * np.exp(-spring_constant / (2 * kt) * (r - r0) ** 2)

    def _numerical_angle_integrand(self, theta, theta0, spring_constant, kt):
        """
        Integrand for the angle (theta) restraints which will be integrated numerically for standard state correction

        This uses a harmonic restraint centered around theta0

        Domain is on [0, pi]

        Parameters
        ----------
        theta : float or np.ndarray of float
            Angle which will be integrated, units of radians
        theta0 : float
            Equilibrium angle at which force of restraint often is 0, units of radians
        spring_constant : float
            Spring constant for this angle in units of kJ/mol/nm**2

        Returns
        -------
        integrand : float
            Value of the integrated
        """
        return np.sin(theta) * np.exp(-spring_constant / (2 * kt) * (theta - theta0) ** 2)

    def _numerical_torsion_integrand(self, phi, phi0, spring_constant, kt):
        """
        Integrand for the torsion (phi) restraints which will be integrated numerically for standard state correction

        Uses a harmonic restraint around phi0 with corrections to the OpenMM limits of dihedral calculation

        Domain is on [-pi, pi], the same domain OpenMM uses

        Parameters
        ----------
        phi : float or np.ndarray of float
            Torsion angle which will be integrated, units of radians
        phi0 : float
            Equilibrium torsion angle at which force of restraint often is 0, units of radians
        spring_constant : float
            Spring constant for this torsion in units of kJ/mol/nm**2
        kt : float
            Boltzmann Temperature of the thermodynamic state restraining the atoms = kB * T
            in units of kJ/mol

        Returns
        -------
        integrand : float
            Value of the integrated
        """
        d_tor = phi - phi0
        dphi = d_tor - np.floor(d_tor / (2 * np.pi) + 0.5) * (2 * np.pi)
        return np.exp(-(spring_constant / 2) / kt * dphi ** 2)
