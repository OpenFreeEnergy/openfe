#!/usr/bin/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Alchemical factory for free energy calculations that operates directly on OpenMM System objects.

DESCRIPTION

This module contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

* `AbsoluteAlchemicalFactory` uses fused elecrostatic and steric alchemical modifications.

"""

# TODO
# - Generalize treatment of nonbonded sterics/electrostatics intra-alchemical
#   forces to support arbitrary mixing rules. Can we eliminate decoupling to something simpler?
# - Add support for other GBSA models.
# - Add functions for the automatic optimization of alchemical states?
# - Can we store serialized form of Force objects so that we can save time in reconstituting
#   Force objects when we make copies?  We can even manipulate the XML representation directly.
# - Allow protocols to automatically be resized to arbitrary number of states, to
#   allow number of states to be enlarged to be an integral multiple of number of GPUs.
# - Finish AMOEBA support.
# - Can alchemically-modified System objects share unmodified Force objects to avoid overhead
#   of duplicating Forces that are not modified?
# - Add support for arbitrary softcore reprogramming of all Custom*Force classes


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================
import copy
import logging
import collections
import itertools
import re

import numpy as np
try:
    import openmm
    from openmm import unit
except ImportError:  # OpenMM < 7.6
    from simtk import openmm, unit

from openmmtools import states, forcefactories, utils
from openmmtools.constants import ONE_4PI_EPS0

logger = logging.getLogger(__name__)


# =============================================================================
# ALCHEMICAL STATE
# =============================================================================

class AlchemicalStateError(states.GlobalParameterError):
    """Error raised by an AlchemicalState."""
    pass


class AlchemicalFunction(states.GlobalParameterFunction):
    """A function of alchemical variables.

    Parameters
    ----------
    expression : str
        A mathematical expression involving alchemical variables.

    Examples
    --------
    >>> alchemical_state = AlchemicalState(lambda_sterics=1.0, lambda_angles=1.0)
    >>> alchemical_state.set_alchemical_variable('lambda', 0.5)
    >>> alchemical_state.set_alchemical_variable('lambda2', 1.0)
    >>> alchemical_state.lambda_sterics = AlchemicalFunction('lambda**2')
    >>> alchemical_state.lambda_sterics
    0.25
    >>> alchemical_state.lambda_angles = AlchemicalFunction('(lambda + lambda2) / 2')
    >>> alchemical_state.lambda_angles
    0.75

    """
    # This class just provides an alternative name to GlobalParameterFunction.
    pass


class AlchemicalState(states.GlobalParameterState):
    """Represent an alchemical state.

    The alchemical parameters modify the Hamiltonian and affect the
    computation of the energy. Alchemical parameters that have value
    None are considered undefined, which means that applying this
    state to System and Context that have that parameter as a global
    variable will raise an AlchemicalStateError.

    Parameters
    ----------
    parameters_name_suffix : str, optional
        If specified, the state will control a modified version of the global
        parameters with the name ``parameter_name + '_' + parameters_name_suffix``.
        When this is the case, the normal parameters are not accessible.
    lambda_sterics : float, optional
        Scaling factor for ligand sterics (Lennard-Jones and Halgren)
        interactions (default is 1.0).
    lambda_electrostatics : float, optional
        Scaling factor for ligand charges, intrinsic Born radii, and surface
        area term (default is 1.0).
    lambda_bonds : float, optional
        Scaling factor for alchemically-softened bonds (default is 1.0).
    lambda_angles : float, optional
        Scaling factor for alchemically-softened angles (default is 1.0).
    lambda_torsions : float, optional
        Scaling factor for alchemically-softened torsions (default is 1.0).

    Attributes
    ----------
    lambda_sterics
    lambda_electrostatics
    lambda_bonds
    lambda_angles
    lambda_torsions

    Examples
    --------
    Create an alchemically modified system.

    >>> from openmmtools import testsystems
    >>> factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
    >>> alanine_vacuum = testsystems.AlanineDipeptideVacuum().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=range(22))
    >>> alanine_alchemical_system = factory.create_alchemical_system(reference_system=alanine_vacuum,
    ...                                                              alchemical_regions=alchemical_region)

    Create a completely undefined alchemical state.

    >>> alchemical_state = AlchemicalState()
    >>> print(alchemical_state.lambda_sterics)
    None
    >>> alchemical_state.apply_to_system(alanine_alchemical_system)
    Traceback (most recent call last):
    ...
    openmmtools.alchemy.AlchemicalStateError: The system parameter lambda_electrostatics is not defined in this state.

    Create an AlchemicalState that matches the parameters defined in
    the System.

    >>> alchemical_state = AlchemicalState.from_system(alanine_alchemical_system)
    >>> alchemical_state.lambda_sterics
    1.0
    >>> alchemical_state.lambda_electrostatics
    1.0
    >>> print(alchemical_state.lambda_angles)
    None

    AlchemicalState implement the IComposableState interface, so it can be
    used with CompoundThermodynamicState. All the alchemical parameters are
    accessible through the compound state.

    >>> import openmm
    >>> from openmm import unit
    >>> thermodynamic_state = states.ThermodynamicState(system=alanine_alchemical_system,
    ...                                                 temperature=300*unit.kelvin)
    >>> compound_state = states.CompoundThermodynamicState(thermodynamic_state=thermodynamic_state,
    ...                                                    composable_states=[alchemical_state])
    >>> compound_state.lambda_sterics
    1.0

    You can control the parameters in the OpenMM Context in this state by
    setting the state attributes.

    >>> compound_state.lambda_sterics = 0.5
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = compound_state.create_context(integrator)
    >>> context.getParameter('lambda_sterics')
    0.5
    >>> compound_state.lambda_sterics = 1.0
    >>> compound_state.apply_to_context(context)
    >>> context.getParameter('lambda_sterics')
    1.0

    You can express the alchemical parameters as a mathematical expression
    involving alchemical variables. Here is an example for a two-stage function.

    >>> compound_state.set_alchemical_variable('lambda', 1.0)
    >>> compound_state.lambda_sterics = AlchemicalFunction('step_hm(lambda - 0.5) + 2*lambda * step_hm(0.5 - lambda)')
    >>> compound_state.lambda_electrostatics = AlchemicalFunction('2*(lambda - 0.5) * step(lambda - 0.5)')
    >>> for l in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ...     compound_state.set_alchemical_variable('lambda', l)
    ...     print(compound_state.lambda_sterics)
    0.0
    0.5
    1.0
    1.0
    1.0

    """

    _GLOBAL_PARAMETER_ERROR = AlchemicalStateError

    # -------------------------------------------------------------------------
    # Lambda properties
    # -------------------------------------------------------------------------

    class _LambdaParameter(states.GlobalParameterState.GlobalParameter):
        """A global parameter in the interval [0, 1] with standard value 1."""

        def __init__(self, parameter_name):
            super().__init__(parameter_name, standard_value=1.0, validator=self.lambda_validator)

        @staticmethod
        def lambda_validator(self, instance, parameter_value):
            if parameter_value is None:
                return parameter_value
            if not (0.0 <= parameter_value <= 1.0):
                raise ValueError('{} must be between 0 and 1.'.format(self.parameter_name))
            return float(parameter_value)

    lambda_sterics_ligandA = _LambdaParameter('lambda_sterics_ligandA')
    lambda_sterics_ligandB = _LambdaParameter('lambda_sterics_ligandB')
    lambda_electrostatics_ligandA = _LambdaParameter('lambda_electrostatics_ligandA')
    lambda_electrostatics_ligandB = _LambdaParameter('lambda_electrostatics_ligandB')
    lambda_restraints_ligandA = _LambdaParameter('lambda_restraints_ligandA')
    lambda_restraints_ligandB = _LambdaParameter('lambda_restraints_ligandB')
    lambda_bonds = _LambdaParameter('lambda_bonds')
    lambda_angles = _LambdaParameter('lambda_angles')
    lambda_torsions = _LambdaParameter('lambda_torsions')

    @classmethod
    def from_system(cls, system, *args, **kwargs):
        """Constructor reading the state from an alchemical system.

        Parameters
        ----------
        system : openmm.System
            An alchemically modified system in a defined alchemical state.
        parameters_name_suffix : str, optional
            If specified, the state will search for a modified
            version of the alchemical parameters with the name
            ``parameter_name + '_' + parameters_name_suffix``.

        Returns
        -------
        The AlchemicalState object representing the alchemical state of
        the system.

        Raises
        ------
        AlchemicalStateError
            If the same parameter has different values in the system, or
            if the system has no lambda parameters.

        """
        # The function is redefined here only to provide more specific documentation for this method.
        return super().from_system(system, *args, **kwargs)

    def set_alchemical_parameters(self, new_value):
        """Set all defined lambda parameters to the given value.

        The undefined parameters (i.e. those being set to None) remain
        undefined.

        Parameters
        ----------
        new_value : float
            The new value for all defined parameters.

        """
        for parameter_name in self._parameters:
            if self._parameters[parameter_name] is not None:
                setattr(self, parameter_name, new_value)

    # -------------------------------------------------------------------------
    # Function variables
    # -------------------------------------------------------------------------

    def get_function_variable(self, variable_name):
        """Return the value of the function variable.

        Function variables are variables entering mathematical expressions
        specified with ``AlchemicalFunction``, which can be use to enslave
        a lambda parameter to arbitrary variables.

        Parameters
        ----------
        variable_name : str
            The name of the function variable.

        Returns
        -------
        variable_value : float
            The value of the function variable.

        """
        # The function is redefined here only to provide more specific documentation for this method.
        return super().get_function_variable(variable_name)

    def set_function_variable(self, variable_name, new_value):
        """Set the value of the function variable.

        Function variables are variables entering mathematical expressions
        specified with ``AlchemicalFunction``, which can be use to enslave
        a lambda parameter to arbitrary variables.

        Parameters
        ----------
        variable_name : str
            The name of the function variable.
        new_value : float
            The new value for the variable.

        """
        # The function is redefined here only to provide more specific documentation for this method.
        super().set_function_variable(variable_name, new_value)

    def get_alchemical_variable(self, variable_name):
        """Return the value of the alchemical parameter.

        .. warning:
            This is deprecated. Use ``get_function_variable`` instead.

        Parameters
        ----------
        variable_name : str
            The name of the alchemical variable.

        Returns
        -------
        variable_value : float
            The value of the alchemical variable.
        """
        import warnings
        warnings.warn('AlchemicalState.get_alchemical_variable is deprecated. '
                      'Use AlchemicalState.get_function_variable instead.')
        return super().get_function_variable(variable_name)

    def set_alchemical_variable(self, variable_name, new_value):
        """Set the value of the alchemical variable.

        .. warning:
            This is deprecated. Use ``set_function_variable`` instead.

        Parameters
        ----------
        variable_name : str
            The name of the alchemical variable.
        new_value : float
            The new value for the variable.

        """
        import warnings
        warnings.warn('AlchemicalState.get_alchemical_variable is deprecated. '
                      'Use AlchemicalState.get_function_variable instead.')
        super().set_function_variable(variable_name, new_value)

    # -------------------------------------------------------------------------
    # IComposableState interface
    # -------------------------------------------------------------------------

    def apply_to_system(self, system):
        """Set the alchemical state of the system to this.

        Parameters
        ----------
        system : openmm.System
            The system to modify.

        Raises
        ------
        AlchemicalStateError
            If the system does not have the required lambda global variables.

        """
        # The function is redefined here only to provide more specific documentation for this method.
        super().apply_to_system(system)

    def check_system_consistency(self, system):
        """Check if the system is in this alchemical state.

        It raises a AlchemicalStateError if the system is not consistent
        with the alchemical state.

        Parameters
        ----------
        system : openmm.System
            The system to test.

        Raises
        ------
        AlchemicalStateError
            If the system is not consistent with this state.

        """
        # The function is redefined here only to provide more specific documentation for this method.
        super().check_system_consistency(system)

    def apply_to_context(self, context):
        """Put the Context into this AlchemicalState.

        Parameters
        ----------
        context : openmm.Context
            The context to set.

        Raises
        ------
        AlchemicalStateError
            If the context does not have the required lambda global variables.

        """
        # The function is redefined here only to provide more specific documentation for this method.
        super().apply_to_context(context)


# =============================================================================
# ALCHEMICAL REGION
# =============================================================================

_ALCHEMICAL_REGION_ARGS = collections.OrderedDict([
    ('alchemical_atoms', None),
    ('alchemical_bonds', None),
    ('alchemical_angles', None),
    ('alchemical_torsions', None),
    ('annihilate_electrostatics', True),
    ('annihilate_sterics', False),
    ('softcore_alpha', 0.5), ('softcore_a', 1), ('softcore_b', 1), ('softcore_c', 6),
    ('softcore_beta', 0.0), ('softcore_d', 1), ('softcore_e', 1), ('softcore_f', 2),
    ('name', None)
])


# The class is just a way to document the namedtuple.
class AlchemicalRegion(collections.namedtuple('AlchemicalRegion', _ALCHEMICAL_REGION_ARGS.keys())):
    """Alchemical region.

    This is a namedtuple used to tell the AbsoluteAlchemicalFactory which
    region of the system to alchemically modify and how.

    Parameters
    ----------
    alchemical_atoms : list of int, optional
        List of atoms to be designated for which the nonbonded forces (both
        sterics and electrostatics components) have to be alchemically
        modified (default is None).
    alchemical_bonds : bool or list of int, optional
        If a list of bond indices are specified, these HarmonicBondForce
        entries are softened with 'lambda_bonds'. If set to True, this list
        is auto-generated to include all bonds involving any alchemical
        atoms (default is None).
    alchemical_angles : bool or list of int, optional
        If a list of angle indices are specified, these HarmonicAngleForce
        entries are softened with 'lambda_angles'. If set to True, this
        list is auto-generated to include all angles involving any alchemical
        atoms (default is None).
    alchemical_torsions : bool or list of int, optional
        If a list of torsion indices are specified, these PeriodicTorsionForce
        entries are softened with 'lambda_torsions'. If set to True, this list
        is auto-generated to include al proper torsions involving any alchemical
        atoms. Improper torsions are not softened (default is None).
    annihilate_electrostatics : bool, optional
        If True, electrostatics should be annihilated, rather than decoupled
        (default is True).
    annihilate_sterics : bool, optional
        If True, sterics (Lennard-Jones or Halgren potential) will be annihilated,
        rather than decoupled (default is False).
    softcore_alpha : float, optional
        Alchemical softcore parameter for Lennard-Jones (default is 0.5).
    softcore_a, softcore_b, softcore_c : float, optional
        Parameters modifying softcore Lennard-Jones form. Introduced in
        Eq. 13 of Ref. [1] (default is 1).
    softcore_beta : float, optional
        Alchemical softcore parameter for electrostatics. Set this to zero
        to recover standard electrostatic scaling (default is 0.0).
    softcore_d, softcore_e, softcore_f : float, optional
        Parameters modifying softcore electrostatics form (default is 1).

    Notes
    -----
    The parameters softcore_e and softcore_f determine the effective distance
    between point charges according to

    r_eff = sigma*((softcore_beta*(lambda_electrostatics-1)^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f)

    References
    ----------
    [1] Pham TT and Shirts MR. Identifying low variance pathways for free
    energy calculations of molecular transformations in solution phase.
    JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

    """
AlchemicalRegion.__new__.__defaults__ = tuple(_ALCHEMICAL_REGION_ARGS.values())


# =============================================================================
# ABSOLUTE ALCHEMICAL FACTORY
# =============================================================================

class AbsoluteAlchemicalFactory(object):
    """Factory of alchemically modified OpenMM Systems.

    The context parameters created are:
    - softcore_alpha: factor controlling softcore lengthscale for Lennard-Jones
    - softcore_beta: factor controlling softcore lengthscale for Coulomb
    - softcore_a: softcore Lennard-Jones parameter from Eq. 13 of Ref [1]
    - softcore_b: softcore Lennard-Jones parameter from Eq. 13 of Ref [1]
    - softcore_c: softcore Lennard-Jones parameter from Eq. 13 of Ref [1]
    - softcore_d: softcore electrostatics parameter
    - softcore_e: softcore electrostatics parameter
    - softcore_f: softcore electrostatics parameter

    Parameters
    ----------
    consistent_exceptions : bool, optional, default = False
        If True, the same functional form of the System's nonbonded
        method will be use to determine the electrostatics contribution
        to the potential energy of 1,4 exceptions instead of the
        classical q1*q2/(4*epsilon*epsilon0*pi*r).
    switch_width : float, optional, default = 1.0 * angstroms
        Default switch width for electrostatics in periodic cutoff systems
        used in alchemical interactions only.
    alchemical_pme_treatment : str, optional, default = 'exact'
        Controls how alchemical region electrostatics are treated when PME is used.
        Options are ['direct-space', 'coulomb', 'exact'].
        - 'direct-space' only models the direct space contribution
        - 'coulomb' includes switched Coulomb interaction
        - 'exact' includes also the reciprocal space contribution, but it's
           only possible to annihilate the charges and the softcore parameters
           controlling the electrostatics are deactivated. Also, with this
           method, modifying the global variable `lambda_electrostatics` is
           not sufficient to control the charges. The recommended way to change
           them is through the `AlchemicalState` class.
    alchemical_rf_treatment : str, optional, default = 'switched'
        Controls how alchemical region electrostatics are treated when RF is used
        Options are ['switched', 'shifted']
        'switched' sets c_rf = 0 for all reaction-field interactions and ensures continuity with a switch
        'shifted' retains c_rf != 0 but can give erroneous results for hydration free energies
    disable_alchemical_dispersion_correction : bool, optional, default=False
        If True, the long-range dispersion correction will not be included for the alchemical
        region to avoid the need to recompute the correction (a CPU operation that takes ~ 0.5 s)
        every time 'lambda_sterics' is changed. If using nonequilibrium protocols, it is recommended
        that this be set to True since this can lead to enormous (100x) slowdowns if the correction
        must be recomputed every time step.
    split_alchemical_forces : bool, optional, default=True
        If True, forces that are altered to different alchemical variables
        will be split in different force groups. All non-alchemical forces
        will maintain their original force group. If more than 32 force
        groups are required, an error is thrown.

    Examples
    --------

    Create an alchemical factory to alchemically modify OpenMM System objects.

    >>> factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)

    Create an alchemically modified version of p-xylene in T4 lysozyme L99A in GBSA.

    >>> # Create a reference system.
    >>> from openmmtools import testsystems
    >>> reference_system = testsystems.LysozymeImplicit().system
    >>> # Alchemically soften the pxylene atoms
    >>> pxylene_atoms = range(2603,2621) # p-xylene
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=pxylene_atoms)
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Alchemically modify one water in a water box.

    >>> reference_system = testsystems.WaterBox().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0, 1, 2])
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Alchemically modify some angles and torsions in alanine dipeptide and
    annihilate both sterics and electrostatics.

    >>> reference_system = testsystems.AlanineDipeptideVacuum().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0], alchemical_torsions=[0,1,2],
    ...                                      alchemical_angles=[0,1,2], annihilate_sterics=True,
    ...                                      annihilate_electrostatics=True)
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Alchemically modify a bond, angles, and torsions in toluene by automatically
    selecting bonds involving alchemical atoms.

    >>> toluene_implicit = testsystems.TolueneImplicit()
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0,1], alchemical_torsions=True,
    ...                                      alchemical_angles=True, annihilate_sterics=True)
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Once the alchemical system is created, you can modify its Hamiltonian
    through AlchemicalState

    >>> alchemical_state = AlchemicalState.from_system(alchemical_system)
    >>> alchemical_state.lambda_sterics
    1.0
    >>> alchemical_state.lambda_electrostatics = 0.5
    >>> alchemical_state.apply_to_system(alchemical_system)

    You can also modify its Hamiltonian directly into a context

    >>> import openmm
    >>> from openmm import unit
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = openmm.Context(alchemical_system, integrator)
    >>> alchemical_state.set_alchemical_parameters(0.0)  # Set all lambda to 0
    >>> alchemical_state.apply_to_context(context)

    Neglecting the long-range dispersion correction for the alchemical region
    (for nonequilibrium switching, for example) requires instantiating a factory
    with the appropriate options:

    >>> new_factory = AbsoluteAlchemicalFactory(consistent_exceptions=False, disable_alchemical_dispersion_correction=True)
    >>> reference_system = testsystems.WaterBox().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0, 1, 2])
    >>> alchemical_system = new_factory.create_alchemical_system(reference_system, alchemical_region)

    References
    ----------
    [1] Pham TT and Shirts MR. Identifying low variance pathways for free
    energy calculations of molecular transformations in solution phase.
    JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

    """

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, consistent_exceptions=False, switch_width=1.0*unit.angstroms,
                 alchemical_pme_treatment='exact', alchemical_rf_treatment='switched',
                 disable_alchemical_dispersion_correction=False, split_alchemical_forces=True):

        self.consistent_exceptions = consistent_exceptions
        self.switch_width = switch_width
        self.alchemical_pme_treatment = alchemical_pme_treatment
        self.alchemical_rf_treatment = alchemical_rf_treatment
        self.disable_alchemical_dispersion_correction = disable_alchemical_dispersion_correction
        self.split_alchemical_forces = split_alchemical_forces

    def create_alchemical_system(self, reference_system, alchemical_regions,
                                 alchemical_regions_interactions=frozenset()):
        """Create an alchemically modified version of the reference system.

        To alter the alchemical state of the returned system use AlchemicalState.

        Parameters
        ----------
        reference_system : openmm.System
            The system to use as a reference for the creation of the
            alchemical system. This will not be modified.
        alchemical_regions : AlchemicalRegion
            The region of the reference system to alchemically soften.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        Returns
        -------
        alchemical_system : openmm.System
            Alchemically-modified version of reference_system.

        """
        if alchemical_regions_interactions != frozenset():
            raise NotImplementedError('Interactions between alchemical regions is untested')

        logger.debug(f'Dictionary of interacting alchemical regions: {alchemical_regions_interactions}')
        if isinstance(alchemical_regions, AlchemicalRegion):
            alchemical_regions = [alchemical_regions]
        logger.debug(f'Using {len(alchemical_regions)} alchemical regions')

        # Resolve alchemical regions.
        alchemical_regions = [self._resolve_alchemical_region(reference_system, alchemical_region)
                              for alchemical_region in alchemical_regions]

        # Check for duplicate alchemical atoms/bonds/angles/torsions.
        all_alchemical_elements = {element_type: set() for element_type in ['atoms', 'bonds', 'angles', 'torsions']}

        for alchemical_region in alchemical_regions:
            for element_type, all_elements in all_alchemical_elements.items():
                # Ignore None alchemical elements.
                region_elements = getattr(alchemical_region, 'alchemical_' + element_type)
                if region_elements is None:
                    continue

                # Check if there are duplicates with previous regions.
                duplicate_elements = all_elements & region_elements
                if len(duplicate_elements) > 0:
                    raise ValueError('Two regions have duplicate {}.'.format(element_type))

                # Update all alchemical elements.
                all_alchemical_elements[element_type].update(region_elements)

        # Check for duplicate names
        alchemical_region_names = {alchemical_region.name for alchemical_region in alchemical_regions}
        if len(alchemical_region_names) != len(alchemical_regions):
            raise ValueError('Two alchemical regions have the same name')

        # Record timing statistics.
        timer = utils.Timer()
        timer.start('Create alchemically modified system')

        # Build alchemical system to modify. This copies particles, vsites,
        # constraints, box vectors and all the forces. We'll later remove
        # the forces that we remodel to be alchemically modified.
        alchemical_system = copy.deepcopy(reference_system)

        # Check that there are no virtual sites to alchemically modify.
        for alchemical_region in alchemical_regions:
            for particle_index in alchemical_region.alchemical_atoms:
                if reference_system.isVirtualSite(particle_index):
                    raise ValueError(f'Virtual atoms in region {alchemical_region.name}. '
                                      'Alchemically modified virtual sites are not supported')

        # Modify forces as appropriate. We delete the forces that
        # have been processed modified at the end of the for loop.
        forces_to_remove = []
        alchemical_forces_by_lambda = {}
        for force_index, reference_force in enumerate(reference_system.getForces()):
            # TODO switch to functools.singledispatch when we drop Python2 support
            reference_force_name = reference_force.__class__.__name__
            alchemical_force_creator_name = '_alchemically_modify_{}'.format(reference_force_name)
            print(alchemical_force_creator_name)
            try:
                alchemical_force_creator_func = getattr(self, alchemical_force_creator_name)
            except AttributeError as e:
                pass
            else:
                # The reference system force will be deleted.
                forces_to_remove.append(force_index)
                # Collect all the Force objects modeling the reference force.
                alchemical_forces = alchemical_force_creator_func(reference_force, alchemical_regions,
                                                                  alchemical_regions_interactions)
                for lambda_variable_name, lambda_forces in alchemical_forces.items():
                    try:
                        alchemical_forces_by_lambda[lambda_variable_name].extend(lambda_forces)
                    except KeyError:
                        alchemical_forces_by_lambda[lambda_variable_name] = lambda_forces

        # Remove original forces that have been alchemically modified.
        print(forces_to_remove)
        for force_index in reversed(forces_to_remove):
            alchemical_system.removeForce(force_index)
        print(alchemical_forces_by_lambda)
        # Add forces and split groups if necessary.
        self._add_alchemical_forces(alchemical_system, alchemical_forces_by_lambda)

        # Record timing statistics.
        timer.stop('Create alchemically modified system')
        timer.report_timing()

        # If the System uses a NonbondedForce, replace its NonbondedForce implementation of reaction field
        # with a Custom*Force implementation that uses c_rf = 0.
        # NOTE: This adds an additional CustomNonbondedForce
        if self.alchemical_rf_treatment == 'switched':
            forcefactories.replace_reaction_field(alchemical_system, return_copy=False,
                                                  switch_width=self.switch_width)

        return alchemical_system

    @classmethod
    def get_energy_components(cls, alchemical_system, alchemical_state, positions, platform=None):
        """Compute potential energy of the alchemical system by Force.

        This can be useful for debug and analysis.

        Parameters
        ----------
        alchemical_system : openmm.AlchemicalSystem
            An alchemically modified system.
        alchemical_state : AlchemicalState
            The alchemical state to set the Context to.
        positions : openmm.unit.Quantity of dimension (natoms, 3)
            Coordinates to use for energy test (units of distance).
        platform : openmm.Platform, optional
            The OpenMM platform to use to compute the energy. If None,
            OpenMM tries to select the fastest available.

        Returns
        -------
        energy_components : dict str: openmm.unit.Quantity
            A string label describing the role of the force associated to
            its contribution to the potential energy.

        """
        # Find and label all forces.
        force_labels = cls._find_force_components(alchemical_system)
        assert len(force_labels) <= 32, ("The alchemical system has more than 32 force groups; "
                                         "can't compute individual force component energies.")

        # Create deep copy of alchemical system.
        system = copy.deepcopy(alchemical_system)

        # Separate all forces into separate force groups.
        for force_index, force in enumerate(system.getForces()):
            force.setForceGroup(force_index)

        # Create a Context in the given state.
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        if platform is None:
            context = openmm.Context(system, integrator)
        else:
            context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        alchemical_state.apply_to_context(context)

        # Get energy components
        energy_components = collections.OrderedDict()
        for force_label, force_index in force_labels.items():
            energy_components[force_label] = context.getState(getEnergy=True,
                                                                groups=2**force_index).getPotentialEnergy()

        # Clean up
        del context, integrator
        return energy_components

    # -------------------------------------------------------------------------
    # Internal usage: AlchemicalRegion
    # -------------------------------------------------------------------------

    @classmethod
    def _resolve_alchemical_region(cls, system, alchemical_region):
        """Return a new AlchemicalRegion with sets of bonds/angles/torsions resolved.

        Also transform any list of indices into a frozenset.

        Parameters
        ----------
        system : openmm.System
            The system.
        alchemical_region : AlchemicalRegion
            The alchemical region of the system.

        Returns
        -------
        AlchemicalRegion
            A new AlchemicalRegion object in which all alchemical_X (where X is
            atoms, bonds, angles, or torsions) have been converted to a frozenset
            of indices that belong to the System.

        Raises
        ------
        ValueError
            If the indices in the AlchemicalRegion can't be found in the system.

        """
        # TODO move to AlchemicalRegion?
        # TODO process also custom forces? (also in _build_alchemical_X_list methods)
        # Find and cache the reference forces.
        reference_forces = {force.__class__.__name__: force for force in system.getForces()}

        # Count number of particles, angles, etc. in system. Atoms
        # must be processed first since the others build on that.
        reference_counts = collections.OrderedDict([
            ('atom', system.getNumParticles()),
            ('bond', reference_forces['HarmonicBondForce'].getNumBonds()
             if 'HarmonicBondForce' in reference_forces else 0),
            ('angle', reference_forces['HarmonicAngleForce'].getNumAngles()
             if 'HarmonicAngleForce' in reference_forces else 0),
            ('torsion', reference_forces['PeriodicTorsionForce'].getNumTorsions()
             if 'PeriodicTorsionForce' in reference_forces else 0)
        ])

        # Transform named tuple to dict for easy modification.
        alchemical_region = alchemical_region._asdict()

        for region in reference_counts:
            region_name = 'alchemical_' + region + 's'
            region_indices = alchemical_region[region_name]

            # Convert None and False to empty lists.
            if region_indices is None or region_indices is False:
                region_indices = set()

            # Automatically build indices list if True.
            elif region_indices is True:
                if reference_counts[region] == 0:
                    region_indices = set()
                else:
                    # TODO switch to functools.singledispatch when drop Python2
                    builder_function = getattr(cls, '_build_alchemical_{}_list'.format(region))
                    region_indices = builder_function(alchemical_region['alchemical_atoms'],
                                                      reference_forces, system)

            # Convert numpy arrays to Python lists since SWIG
            # have problems with np.int (see openmm#1650).
            elif isinstance(region_indices, np.ndarray):
                region_indices = region_indices.tolist()

            # Convert to set and update alchemical region.
            region_indices = frozenset(region_indices)
            alchemical_region[region_name] = region_indices

            # Check that the given indices are in the system.
            indices_diff = region_indices - set(range(reference_counts[region]))
            if len(indices_diff) > 0:
                err_msg = 'Indices {} in {} cannot be found in the system'
                raise ValueError(err_msg.format(indices_diff, region_name))

        # Check that an alchemical region is defined.
        total_alchemically_modified = 0
        for region in reference_counts:
            total_alchemically_modified += len(alchemical_region['alchemical_' + region + 's'])
        if total_alchemically_modified == 0:
            raise ValueError('The AlchemicalRegion is empty.')

        # Return a new AlchemicalRegion with the resolved indices lists.
        return AlchemicalRegion(**alchemical_region)

    @staticmethod
    def _tabulate_bonds(system):
        """
        Tabulate bonds for the specified system.

        Parameters
        ----------
        system : openmm.System
            The system for which bonds are to be tabulated.

        Returns
        -------
        bonds : list of set
            bonds[i] is the set of bonds to atom i

        TODO:
        * Could we use a Topology object to simplify this?

        """
        bonds = [set() for _ in range(system.getNumParticles())]

        forces = {system.getForce(index).__class__.__name__: system.getForce(index)
                  for index in range(system.getNumForces())}

        # Process HarmonicBondForce
        bond_force = forces['HarmonicBondForce']
        for bond_index in range(bond_force.getNumBonds()):
            [particle1, particle2, r, K] = bond_force.getBondParameters(bond_index)
            bonds[particle1].add(particle2)
            bonds[particle2].add(particle1)
        # Process constraints.
        for constraint_index in range(system.getNumConstraints()):
            [particle1, particle2, r] = system.getConstraintParameters(constraint_index)
            bonds[particle1].add(particle2)
            bonds[particle2].add(particle1)

        # TODO: Process CustomBondForce?

        return bonds

    @classmethod
    def _build_alchemical_torsion_list(cls, alchemical_atoms, reference_forces, system):
        """
        Build a list of proper torsion indices that involve any alchemical atom.

        Parameters
        ----------
        alchemical_atoms : set of int
            The set of alchemically modified atoms.
        reference_forces : dict str: force
            A dictionary of cached forces in the system accessible by names.
        system : openmm.System
            The system.

        Returns
        -------
        torsion_list : list of int
            The list of torsion indices that should be alchemically softened

        """

        # Tabulate all bonds
        bonds = cls._tabulate_bonds(system)

        def is_bonded(i, j):
            if j in bonds[i]:
                return True
            return False

        def is_proper_torsion(i, j, k, l):
            if is_bonded(i, j) and is_bonded(j, k) and is_bonded(k, l):
                return True
            return False

        # Create a list of proper torsions that involve any alchemical atom.
        torsion_list = list()
        force = reference_forces['PeriodicTorsionForce']
        for torsion_index in range(force.getNumTorsions()):
            particle1, particle2, particle3, particle4, periodicity, phase, k = force.getTorsionParameters(torsion_index)
            if set([particle1, particle2, particle3, particle4]).intersection(alchemical_atoms):
                if is_proper_torsion(particle1, particle2, particle3, particle4):
                    torsion_list.append(torsion_index)

        return torsion_list

    @staticmethod
    def _build_alchemical_angle_list(alchemical_atoms, reference_forces, system):
        """
        Build a list of angle indices that involve any alchemical atom.

        Parameters
        ----------
        alchemical_atoms : set of int
            The set of alchemically modified atoms.
        reference_forces : dict str: force
            A dictionary of cached forces in the system accessible by names.
        system : openmm.System
            The system (unused).

        Returns
        -------
        angle_list : list of int
            The list of angle indices that should be alchemically softened

        """
        angle_list = list()
        force = reference_forces['HarmonicAngleForce']
        for angle_index in range(force.getNumAngles()):
            [particle1, particle2, particle3, theta0, K] = force.getAngleParameters(angle_index)
            if set([particle1, particle2, particle3]).intersection(alchemical_atoms):
                angle_list.append(angle_index)

        return angle_list

    @staticmethod
    def _build_alchemical_bond_list(alchemical_atoms, reference_forces, system):
        """
        Build a list of bond indices that involve any alchemical atom, allowing a list of bonds to override.

        Parameters
        ----------
        alchemical_atoms : set of int
            The set of alchemically modified atoms.
        reference_forces : dict str: force
            A dictionary of cached forces in the system accessible by names.
        system : openmm.System
            The system (unused).

        Returns
        -------
        bond_list : list of int
            The list of bond indices that should be alchemically softened

        """
        bond_list = list()
        force = reference_forces['HarmonicBondForce']
        for bond_index in range(force.getNumBonds()):
            [particle1, particle2, r, K] = force.getBondParameters(bond_index)
            if set([particle1, particle2]).intersection(alchemical_atoms):
                bond_list.append(bond_index)

        return bond_list

    # -------------------------------------------------------------------------
    # Internal usage: Alchemical forces
    # -------------------------------------------------------------------------

    def _add_alchemical_forces(self, alchemical_system, alchemical_forces_by_lambda):
        """Add the forces to the alchemical system and eventually split the force groups."""
        # OpenMM can have a maximum of 32 groups.
        available_force_groups = set(range(32))

        # Add non-alchemical groups. New forces will have force group 0, and we don't
        # want to modify the force group of forces that have been copied from the reference.
        non_alchemical_forces = alchemical_forces_by_lambda.pop('', [])
        for non_alchemical_force in non_alchemical_forces:
            alchemical_system.addForce(non_alchemical_force)

        # Find which force groups are still available for alchemical forces.
        for force in alchemical_system.getForces():
            available_force_groups.discard(force.getForceGroup())

        # Check if there are enough force groups to split alchemical forces.
        if (self.split_alchemical_forces and
                    len(available_force_groups) < len(alchemical_forces_by_lambda)):
            raise RuntimeError('There are not enough force groups to split alchemical forces.\n'
                               'Consider merging some non-alchemical forces in a single group '
                               'or set split_alchemical_forces to False.')

        # Add the alchemical forces in a deterministic way (just to be safe).
        for lambda_variable in sorted(alchemical_forces_by_lambda):
            if self.split_alchemical_forces:
                # Assign to these forces the smallest force group index available.
                force_group = min(available_force_groups)
                available_force_groups.remove(force_group)
            for force in alchemical_forces_by_lambda[lambda_variable]:
                if self.split_alchemical_forces:
                    force.setForceGroup(force_group)
                alchemical_system.addForce(force)

    @staticmethod
    def _are_straddling_noninteracting_regions(particles, alchemical_regions, alchemical_regions_interactions):
        """Test if a set of particles are in two regions simultaneously and if these regions are interacting.

        Parameters
        ----------
        particles: set
            Set of particles to check if they are in two noninteracting alchemical regions.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the torsions to test.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        """
        for i, alchemical_region_A in enumerate(alchemical_regions):
            if alchemical_region_A.alchemical_atoms.intersection(particles):
                j = 0
                while j < i:
                    alchemical_region_B = alchemical_regions[j]
                    if alchemical_region_B.alchemical_atoms.intersection(particles):
                        if {i, j} in alchemical_regions_interactions:
                            return False
                        else:
                            return True
                    j += 1
        return False

    @classmethod
    def _alchemically_modify_PeriodicTorsionForce(cls, reference_force, alchemical_regions, alchemical_regions_interactions):
        """Create alchemically-modified version of PeriodicTorsionForce.

        Parameters
        ----------
        reference_force : openmm.PeriodicTorsionForce
            The reference PeriodicTorsionForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the torsions to
            alchemically modify.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        Returns
        -------
        force : openmm.PeriodicTorsionForce
            The force responsible for the non-alchemical torsions.
        custom_force : openmm.CustomTorsionForce
            The force responsible for the alchemically-softened torsions.
            This will not be present if there are not alchemical torsions
            in alchemical_region.

        """
        alchemical_forces = {}
        all_alchemical_torsions = set()

        # Don't create a force if there are no alchemical torsions.
        for alchemical_region in alchemical_regions:
            all_alchemical_torsions.update(alchemical_region.alchemical_torsions)
        if len(all_alchemical_torsions) == 0:
            return {'': [copy.deepcopy(reference_force)]}

        # Create PeriodicTorsionForce to handle unmodified torsions.
        force = openmm.PeriodicTorsionForce()
        force.setForceGroup(reference_force.getForceGroup())
        for torsion_index in range(reference_force.getNumTorsions()):
            particle1, particle2, particle3, particle4, periodicity, phase, k = reference_force.getTorsionParameters(
                torsion_index)
            particles = set((particle1, particle2, particle3, particle4))

            # We don't connect through a torsion two particles that belong
            # to two different and noninteracting alchemical regions.
            are_straddling = cls._are_straddling_noninteracting_regions(
                particles, alchemical_regions, alchemical_regions_interactions)
            if (torsion_index not in all_alchemical_torsions) and (not are_straddling):
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)

        # Update the returned value with the non-alchemical force.
        alchemical_forces[''] = [force]

        # Create CustomTorsionForce to handle alchemically modified torsions.
        for alchemical_region in alchemical_regions:
            # This region may not have torsions to modify.
            if len(alchemical_region.alchemical_torsions) == 0:
                continue

            # Check if the lambda variable needs a suffix to identify the region.
            lambda_variable_name = 'lambda_torsions'
            if alchemical_region.name is not None:
                lambda_variable_name += '_' + alchemical_region.name

            # Create CustomTorsionForce to handle alchemically modified torsions.
            energy_function = f"{lambda_variable_name}*k*(1+cos(periodicity*theta-phase))"
            custom_force = openmm.CustomTorsionForce(energy_function)
            custom_force.addGlobalParameter(lambda_variable_name, 1.0)
            custom_force.addPerTorsionParameter('periodicity')
            custom_force.addPerTorsionParameter('phase')
            custom_force.addPerTorsionParameter('k')

            # Process reference torsions.
            for torsion_index in sorted(alchemical_region.alchemical_torsions):
                # Retrieve parameters.
                particle1, particle2, particle3, particle4, periodicity, phase, k = reference_force.getTorsionParameters(torsion_index)
                # Create torsions.
                custom_force.addTorsion(particle1, particle2, particle3, particle4, [periodicity, phase, k])

            # Update the returned value with the alchemical force.
            alchemical_forces.update({lambda_variable_name: [custom_force]})

        return alchemical_forces

    @classmethod
    def _alchemically_modify_HarmonicAngleForce(cls, reference_force, alchemical_regions, alchemical_regions_interactions):
        """Create alchemically-modified version of HarmonicAngleForce

        Parameters
        ----------
        reference_force : openmm.HarmonicAngleForce
            The reference HarmonicAngleForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the angles to
            alchemically modify.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        Returns
        -------
        force : openmm.HarmonicAngleForce
            The force responsible for the non-alchemical angles.
        custom_force : openmm.CustomAngleForce
            The force responsible for the alchemically-softened angles.
            This will not be present if there are not alchemical angles
            in alchemical_region.

        """
        alchemical_forces = {}
        all_alchemical_angles = set()
        # Don't create a force if there are no alchemical angles.
        for alchemical_region in alchemical_regions:
            all_alchemical_angles.update(alchemical_region.alchemical_angles)
        if len(all_alchemical_angles) == 0:
            return {'': [copy.deepcopy(reference_force)]}

        # Create standard HarmonicAngleForce to handle unmodified angles.
        force = openmm.HarmonicAngleForce()
        force.setForceGroup(reference_force.getForceGroup())
        for angle_index in range(reference_force.getNumAngles()):
            [particle1, particle2, particle3, theta0, K] = reference_force.getAngleParameters(angle_index)
            particles = set((particle1, particle2, particle3))

            # We don't connect through an angle two particles that belong
            # to two different and noninteracting alchemical regions.
            are_straddling = cls._are_straddling_noninteracting_regions(
                particles, alchemical_regions, alchemical_regions_interactions)
            if (angle_index not in all_alchemical_angles) and (not are_straddling):
                force.addAngle(particle1, particle2, particle3, theta0, K)

        # Update the returned value with the non-alchemical force.
        alchemical_forces[''] = [force]

        # Create CustomAngleForce to handle alchemically modified angles.
        for alchemical_region in alchemical_regions:
            # This region may not have angles to modify.
            if len(alchemical_region.alchemical_angles) == 0:
                continue

            # Check if the lambda variable needs a suffix to identify the region.
            lambda_variable_name = 'lambda_angles'
            if alchemical_region.name is not None:
                lambda_variable_name += '_' + alchemical_region.name

            # Create CustomAngleForce to handle alchemically modified angles.
            energy_function = f"{lambda_variable_name}*(K/2)*(theta-theta0)^2;"
            custom_force = openmm.CustomAngleForce(energy_function)
            custom_force.addGlobalParameter(lambda_variable_name, 1.0)
            custom_force.addPerAngleParameter('theta0')
            custom_force.addPerAngleParameter('K')

            # Process reference angles.
            for angle_index in sorted(alchemical_region.alchemical_angles):
                [particle1, particle2, particle3, theta0, K] = reference_force.getAngleParameters(angle_index)
                custom_force.addAngle(particle1, particle2, particle3, [theta0, K])

            # Update the returned value with the alchemical force.
            alchemical_forces.update({lambda_variable_name: [custom_force]})

        return alchemical_forces

    @classmethod
    def _alchemically_modify_HarmonicBondForce(cls, reference_force, alchemical_regions, alchemical_regions_interactions):
        """Create alchemically-modified version of HarmonicBondForce

        Parameters
        ----------
        reference_force : openmm.HarmonicBondForce
            The reference HarmonicBondForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the bonds to
            alchemically modify.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        Returns
        -------
        force : openmm.HarmonicBondForce
            The force responsible for the non-alchemical bonds.
        custom_force : openmm.CustomBondForce
            The force responsible for the alchemically-softened bonds.
            This will not be present if there are not alchemical bonds
            in alchemical_region.

        """
        alchemical_forces = {}
        all_alchemical_bonds = set()

        # Don't create a force if there are no alchemical bonds.
        for alchemical_region in alchemical_regions:
               all_alchemical_bonds.update(alchemical_region.alchemical_bonds)
        if len(all_alchemical_bonds) == 0:
            return {'': [copy.deepcopy(reference_force)]}

        # Create standard HarmonicBondForce to handle unmodified bonds.
        force = openmm.HarmonicBondForce()
        force.setForceGroup(reference_force.getForceGroup())
        for bond_index in range(reference_force.getNumBonds()):
            [particle1, particle2, theta0, K] = reference_force.getBondParameters(bond_index)
            particles = set((particle1, particle2))

            # We don't connect through a bond two particles that belong
            # to two different and noninteracting alchemical regions.
            are_straddling = cls._are_straddling_noninteracting_regions(
                particles, alchemical_regions, alchemical_regions_interactions)
            if (bond_index not in all_alchemical_bonds) and (not are_straddling):
                force.addBond(particle1, particle2, theta0, K)

        # Update the returned value with the non-alchemical force.
        alchemical_forces[''] = [force]

        for alchemical_region in alchemical_regions:
            # This region may not have bonds to modify.
            if len(alchemical_region.alchemical_bonds) == 0:
                continue

            # Check if the lambda variable needs a suffix to identify the region.
            lambda_variable_name = 'lambda_bonds'
            if alchemical_region.name is not None:
                lambda_variable_name += '_' + alchemical_region.name

            # Define force here so that it is over writen saving only one copy.
            # Create CustomBondForce to handle alchemically modified bonds.
            energy_function = f"{lambda_variable_name}*(K/2)*(r-r0)^2;"
            custom_force = openmm.CustomBondForce(energy_function)
            custom_force.addGlobalParameter(lambda_variable_name, 1.0)
            custom_force.addPerBondParameter('r0')
            custom_force.addPerBondParameter('K')
            # Process reference bonds.
            for bond_index in sorted(alchemical_region.alchemical_bonds):
                [particle1, particle2, theta0, K] = reference_force.getBondParameters(bond_index)
                custom_force.addBond(particle1, particle2, [theta0, K])

            # Update the returned value with the alchemical force.
            alchemical_forces.update({lambda_variable_name: [custom_force]})

        return alchemical_forces

    def _get_sterics_energy_expressions(self, lambda_variable_suffixes):
        """Return the energy expressions for sterics.

        Parameters
        ----------
        lambda_variable_suffixes : List[str]
            A list with suffixes for the global variable "lambda_sterics" that
            will control the energy. If no suffix is necessary (i.e. there are
            no multiple alchemical regions) just set lambda_variable_suffixes[0] = ''.
            If the list has more than one element, the energy is controlled by
            the multiplication of lambda_sterics_suffix1 * lambda_sterics_suffix2.
        """

        # Sterics mixing rules.
        if lambda_variable_suffixes[0] == '':
            lambda_variable_name = 'lambda_sterics'
        else:
            if len(lambda_variable_suffixes) > 1:
                lambda_variable_name = 'lambda_sterics{0}*lambda_sterics{1}'.format(
                    lambda_variable_suffixes[0], lambda_variable_suffixes[1])
            else:
                lambda_variable_name = 'lambda_sterics{}'.format(lambda_variable_suffixes[0])

        sterics_mixing_rules = ('epsilon = sqrt(epsilon1*epsilon2);'  # Mixing rule for epsilon.
                                'sigma = 0.5*(sigma1 + sigma2);')  # Mixing rule for sigma.

        # Soft-core Lennard-Jones.
        exceptions_sterics_energy_expression = ('U_sterics;'
                                                'U_sterics = (({0})^softcore_a)*4*epsilon*x*(x-1.0);'
                                                'x = (sigma/reff_sterics)^6;'
                                                # Effective softcore distance for sterics.
                                                'reff_sterics = sigma*((softcore_alpha*(1.0-({0}))^softcore_b + (r/sigma)^softcore_c))^(1/softcore_c);')\
                                                .format(lambda_variable_name)
        # Define energy expression for electrostatics.
        return sterics_mixing_rules, exceptions_sterics_energy_expression

    def _get_electrostatics_energy_expressions(self, reference_force, lambda_variable_suffixes):
        """Return the energy expressions for electrostatics.

        This private function assumes self._alchemical_pme_treatment != 'exact'
        as there's no electrostatics CustomNondondedForce in this case, and
        lambda_electrostatics is modeled through an offset parameter in a
        NonbondedForce.

        Parameters
        ----------
        lambda_variable_suffixes : List[str]
            A list with suffixes for the global variable "lambda_electrostatics" that
            will control the energy. If no suffix is necessary (i.e. there are
            no multiple alchemical regions) just set lambda_variable_suffixes[0] = ''.
            If the list has more than one element, the energy is controlled by
            the multiplication of lambda_electrostatics_suffix1 * lambda_electrostatics_suffix2.
        """
        if lambda_variable_suffixes[0] == '':
            lambda_variable_name = 'lambda_electrostatics'
        else:
            if len(lambda_variable_suffixes) > 1:
                lambda_variable_name = 'lambda_electrostatics{0}*lambda_electrostatics{1}'.format(
                    lambda_variable_suffixes[0], lambda_variable_suffixes[1])
            else:
                lambda_variable_name = 'lambda_electrostatics{}'.format(lambda_variable_suffixes[0])

        # The final expression will be prefix + method + suffix.
        electrostatics_prefix = ('U_electrostatics;'
                                     'U_electrostatics=(({})^softcore_d)*ONE_4PI_EPS0*chargeprod').format(lambda_variable_name)

        # Effective softcore distance for electrostatics (common to all methods).
        electrostatics_suffix = ('reff_electrostatics = sigma*((softcore_beta*(1.0-({0}))^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f);'
                                 'ONE_4PI_EPS0 = {1};').format(lambda_variable_name, ONE_4PI_EPS0)  # Already in OpenMM units.

        # Define mixing rules.
        electrostatics_mixing_rules = ('chargeprod = charge1*charge2;'  # Mixing rule for charges.
                                       'sigma = 0.5*(sigma1 + sigma2);')  # Mixing rule for sigma.

        # Standard Coulomb expression with softened core. This is used
        #   - When the nonbonded method of the reference force is NoCutoff.
        #   - When alchemical_pme_treatment is set to 'coulomb'.
        #   - With 1-4 exceptions, unless self.consistent_exceptions is True.
        coulomb_expression = '/reff_electrostatics;'

        # Select electrostatics functional form based on nonbonded method.
        nonbonded_method = reference_force.getNonbondedMethod()

        # Soft-core Coulomb.
        if nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            electrostatics_method_expression = coulomb_expression
        # Reaction-field electrostatics.
        elif nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            electrostatics_method_expression = self._get_reaction_field_unique_expression(reference_force)
        # PME electrostatics.
        elif nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            # Ewald direct-space electrostatics.
            if self.alchemical_pme_treatment == 'direct-space':
                electrostatics_method_expression = self._get_pme_direct_space_unique_expression(reference_force)
            # Use switched standard Coulomb potential, following MTS scheme described in
            # http://dx.doi.org/10.1063/1.1385159
            elif self.alchemical_pme_treatment == 'coulomb':
                electrostatics_method_expression = coulomb_expression
            else:
                raise ValueError("Unknown alchemical_pme_treatment scheme '{}'".format(self.alchemical_pme_treatment))
        else:
            raise ValueError("Nonbonded method {} not supported yet.".format(nonbonded_method))

        # Define energy expression for 1,4 electrostatic exceptions.
        exceptions_electrostatics_energy_expression = electrostatics_prefix
        if self.consistent_exceptions:
            exceptions_electrostatics_energy_expression += electrostatics_method_expression
        else:
            exceptions_electrostatics_energy_expression += coulomb_expression
        exceptions_electrostatics_energy_expression += electrostatics_suffix

        # Define energy expression for electrostatics.
        electrostatics_energy_expression = (electrostatics_prefix + electrostatics_method_expression +
                                            electrostatics_suffix + electrostatics_mixing_rules)

        return electrostatics_energy_expression, exceptions_electrostatics_energy_expression

    def _get_reaction_field_unique_expression(self, reference_force):
        """Unique part of the expression for reaction-field electrostatics.

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            The reference force including the reaction-field parameters.

        Returns
        -------
        rf_expression : str
            The unique expression for reaction-field electrostatics.

        See Also
        --------
        _get_nonbonded_energy_expressions
        """
        epsilon_solvent = reference_force.getReactionFieldDielectric()
        r_cutoff = reference_force.getCutoffDistance()

        # Determine reaction fields parameters.
        k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
        if self.alchemical_rf_treatment == 'switched':
            c_rf = 0.0 / unit.nanometers
        elif self.alchemical_rf_treatment == 'shifted':
            # WARNING: Setting c_rf != 0 can cause large errors in DeltaG for hydration free energies
            c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
        else:
            raise ValueError("Unknown alchemical_rf_treatment scheme '{}'".format(self.alchemical_rf_treatment))

        k_rf = k_rf.value_in_unit_system(unit.md_unit_system)
        c_rf = c_rf.value_in_unit_system(unit.md_unit_system)
        rf_expression = ('*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);'
                         'k_rf = {k_rf};'
                         'c_rf = {c_rf};').format(k_rf=k_rf, c_rf=c_rf)
        return rf_expression

    def _get_pme_direct_space_unique_expression(self, reference_force):
        """Unique part of the expression for Ewald direct-space electrostatics.

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            The reference force including the Ewald parameters.

        Returns
        -------
        rf_expression : str
            The unique expression for Ewald direct-space electrostatics.

        See Also
        --------
        _get_nonbonded_energy_expressions
        """
        # Determine PME parameters.
        [alpha_ewald, nx, ny, nz] = reference_force.getPMEParameters()
        if (alpha_ewald/alpha_ewald.unit) == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
            tol = reference_force.getEwaldErrorTolerance()
            alpha_ewald = (1.0/reference_force.getCutoffDistance()) * np.sqrt(-np.log(2.0*tol))

        alpha_ewald = alpha_ewald.value_in_unit_system(unit.md_unit_system)
        pme_expression = ("*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
                          "alpha_ewald = {};").format(alpha_ewald)
        return pme_expression

    def _alchemically_modify_NonbondedForce(self, reference_force, alchemical_regions, alchemical_regions_interactions):
        """Create alchemically-modified version of NonbondedForce.

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            The reference NonbondedForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the atoms to
            alchemically modify.
        alchemical_regions_interactions : Set[Tuple[int, int]], optional
            Set of alchemical region index pairs for interacting regions.
            By default, all alchemical regions interact only with the
            non-alchemical environment.

        Returns
        -------
        nonbonded_force : openmm.NonbondedForce
            The force responsible for interactions and exceptions of non-alchemical atoms.
        aa_sterics_custom_nonbonded_force : openmm.CustomNonbondedForce
            The force responsible for sterics interactions of alchemical/alchemical atoms.
        aa_electrostatics_custom_nonbonded_force : openmm.CustomNonbondedForce
            The force responsible for electrostatics interactions of alchemical/alchemical
            atoms.
        na_sterics_custom_nonbonded_force : openmm.CustomNonbondedForce
            The force responsible for sterics interactions of non-alchemical/alchemical atoms.
        na_electrostatics_custom_nonbonded_force : openmm.CustomNonbondedForce
            The force responsible for electrostatics interactions of non-alchemical/alchemical
            atoms.
        aa_sterics_custom_bond_force : openmm.CustomBondForce
            The force responsible for sterics exceptions of alchemical/alchemical atoms.
        aa_electrostatics_custom_bond_force : openmm.CustomBondForce
            The force responsible for electrostatics exceptions of alchemical/alchemical
            atoms.
        na_sterics_custom_bond_force : openmm.CustomBondForce
            The force responsible for sterics exceptions of non-alchemical/alchemical atoms.
        na_electrostatics_custom_bond_force : openmm.CustomBondForce
            The force responsible for electrostatics exceptions of non-alchemical/alchemical
            atoms.

        References
        ----------
        [1] Pham TT and Shirts MR. Identifying low variance pathways for free
        energy calculations of molecular transformations in solution phase.
        JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

        """
        # TODO Change softcore_beta to a dimensionless scalar to multiply some intrinsic length-scale, like Lennard-Jones alpha.
        # TODO Try using a single, common "reff" effective softcore distance for both Lennard-Jones and Coulomb.

        forces_by_lambda = {}
        all_alchemical_atoms = set()

        # Don't create a force if there are no alchemical atoms.
        for alchemical_region in alchemical_regions:
            if not len(alchemical_region.alchemical_atoms) == 0:
                all_alchemical_atoms.update(alchemical_region.alchemical_atoms)

        # Don't create a force if there are no alchemical atoms.
        if len(all_alchemical_atoms) == 0:
            return {'': [copy.deepcopy(reference_force)]}

        # Create a set of all the non-alchemical atoms only.
        all_atomset = set(range(reference_force.getNumParticles()))
        nonalchemical_atomset = all_atomset.difference(all_alchemical_atoms)

        # -------------------------------------------------------------
        # Perform tasks that do not need to be repeated for all regions.
        # -------------------------------------------------------------

        # Define energy expression for electrostatics based on nonbonded method.
        nonbonded_method = reference_force.getNonbondedMethod()
        is_ewald_method = nonbonded_method in [openmm.NonbondedForce.Ewald,
                                               openmm.NonbondedForce.PME]
        is_rf_method = nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic,
                                            openmm.NonbondedForce.CutoffNonPeriodic]
        is_periodic_method = is_ewald_method or nonbonded_method == openmm.NonbondedForce.CutoffPeriodic
        use_exact_pme_treatment = is_ewald_method and self.alchemical_pme_treatment == 'exact'

        # Warn about reaction field.
        if is_rf_method:
            logger.warning('Reaction field support is still experimental. For free energy '
                           'calculations in explicit solvent, we suggest using PME for now.')

        # Check that PME treatment is supported with the region's parameters.
        if use_exact_pme_treatment:
            for alchemical_region in alchemical_regions:
                err_msg = ' not supported with exact treatment of Ewald electrostatics.'
                if not alchemical_region.annihilate_electrostatics:
                    raise ValueError('Decoupled electrostatics is' + err_msg)
                if self.consistent_exceptions:
                    raise ValueError('Consistent exceptions are' + err_msg)
                if (alchemical_region.softcore_beta, alchemical_region.softcore_d, alchemical_region.softcore_e) != (0, 1, 1):
                    raise ValueError('Softcore electrostatics is' + err_msg)

        # Create a copy of the NonbondedForce to handle particle interactions and
        # 1,4 exceptions between non-alchemical/non-alchemical atoms (nn).
        nonbonded_force = copy.deepcopy(reference_force)

        # Fix any issues in reference force with Lennard-Jones sigma = 0 (epsilon = 0),
        # which should have sigma > 0.
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
            # Check particle sigma is not zero.
            if sigma == 0.0 * unit.angstrom:
                warning_msg = 'particle %d has Lennard-Jones sigma = 0 (charge=%s, sigma=%s, epsilon=%s); setting sigma=1A'
                logger.warning(warning_msg % (particle_index, str(charge), str(sigma), str(epsilon)))
                sigma = 1.0 * unit.angstrom
                # Fix it.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)

        # Same for the exceptions.
        for exception_index in range(reference_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
            # Check particle sigma is not zero.
            if sigma == 0.0 * unit.angstrom:
                warning_msg = 'exception %d has Lennard-Jones sigma = 0 (iatom=%d, jatom=%d, chargeprod=%s, sigma=%s, epsilon=%s); setting sigma=1A'
                logger.warning(warning_msg % (exception_index, iatom, jatom, str(chargeprod), str(sigma), str(epsilon)))
                sigma = 1.0 * unit.angstrom
                # Fix it.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)

        if use_exact_pme_treatment:
            # Exclude noninteracting alchemical regions from seeing each other in the nonbonded
            for x, y in itertools.combinations(range(len(alchemical_regions)), 2):
                if (x, y) not in alchemical_regions_interactions:
                    for atom1 in alchemical_regions[x].alchemical_atoms:
                        for atom2 in alchemical_regions[y].alchemical_atoms:
                            nonbonded_force.addException(atom1, atom2, 0.0, 1.0, 0.0, True)
                else:
                    region_names = (alchemical_regions[x].name, alchemical_regions[y].name)
                    logger.debug(f'Adding a exact PME electrostatic interaction group between groups {region_names}.')
                    del region_names

            # With exact PME treatment, particle electrostatics is handled through offset parameters.
            for alchemical_region in alchemical_regions:
                if alchemical_region.name is None:
                    nonbonded_force.addGlobalParameter('lambda_electrostatics', 1.0)
                else:
                    nonbonded_force.addGlobalParameter(f'lambda_electrostatics_{alchemical_region.name}', 1.0)

        # Make of list of all single and double permutations of alchemical regions.
        single_regions = [[alchemical_region] for alchemical_region in alchemical_regions]
        if len(alchemical_regions_interactions) == 0:
            pair_regions = []
        else:
            # Only generate pairs of alchemical regions specified by alchemical_regions_interactions.
            pair_regions = [[alchemical_regions[x[0]], alchemical_regions[x[1]]] for x in
                            alchemical_regions_interactions]

        # Iterate over all single and double permutations of alchemical regions to build all interactions.
        for alchemical_regions_pairs in single_regions+pair_regions:
            # Make a list of region names for the alchemical regions interactions which are being built.
            lambda_var_suffixes = []
            for alchemical_region in alchemical_regions_pairs:
                if alchemical_region.name is None:
                    lambda_var_suffixes.append('')
                else:
                    lambda_var_suffixes.append('_' + alchemical_region.name)

            # --------------------------------------------------
            # Determine energy expression for all custom forces
            # --------------------------------------------------

            # Get steric energy expressions.
            sterics_mixing_rules, exceptions_sterics_energy_expression = self._get_sterics_energy_expressions(lambda_var_suffixes)

            # Define energy expression for sterics.
            sterics_energy_expression = exceptions_sterics_energy_expression + sterics_mixing_rules

            if not use_exact_pme_treatment:
                # There's no CustomNonbondedForce that models electrostatics if we use exact
                # PME treatment. Electrostatics is modeled through offset parameters.
                energy_expressions = self._get_electrostatics_energy_expressions(reference_force, lambda_var_suffixes)
                (electrostatics_energy_expression,
                 exceptions_electrostatics_energy_expression) = energy_expressions  # Unpack tuple.

            # ------------------------------------------------------------
            # Create and configure all forces to add to alchemical system
            # ------------------------------------------------------------

            # Interactions and exceptions will be distributed according to the following table.

            # --------------------------------------------------------------------------------------------------
            # FORCE                                    | INTERACTION GROUP                                     |
            # --------------------------------------------------------------------------------------------------
            # nonbonded_force (unmodified)             | all interactions nonalchemical/nonalchemical          |
            #                                          | all exceptions nonalchemical/nonalchemical            |
            # --------------------------------------------------------------------------------------------------
            # aa_sterics_custom_nonbonded_force        | sterics interactions alchemical/alchemical            |
            # --------------------------------------------------------------------------------------------------
            # aa_electrostatics_custom_nonbonded_force | electrostatics interactions alchemical/alchemical     |
            #                                          | (only without exact PME treatment)                    |
            # --------------------------------------------------------------------------------------------------
            # na_sterics_custom_nonbonded_force        | sterics interactions non-alchemical/alchemical        |
            # --------------------------------------------------------------------------------------------------
            # na_electrostatics_custom_nonbonded_force | electrostatics interactions non-alchemical/alchemical |
            #                                          | (only without exact PME treatment)                    |
            # --------------------------------------------------------------------------------------------------
            # aa_sterics_custom_bond_force             | sterics exceptions alchemical/alchemical              |
            # --------------------------------------------------------------------------------------------------
            # aa_electrostatics_custom_bond_force      | electrostatics exceptions alchemical/alchemical       |
            #                                          | (only without exact PME treatment)                    |
            # --------------------------------------------------------------------------------------------------
            # na_sterics_custom_bond_force             | sterics exceptions non-alchemical/alchemical          |
            # --------------------------------------------------------------------------------------------------
            # na_electrostatics_custom_bond_force      | electrostatics exceptions non-alchemical/alchemical   |
            #                                          | (only without exact PME treatment)                    |
            # --------------------------------------------------------------------------------------------------

            def create_force(force_cls, energy_expression, lambda_variable_name, lambda_var_suffixes, is_lambda_controlled):
                """Shortcut to create a lambda-controlled custom forces."""
                if is_lambda_controlled:
                    force = force_cls(energy_expression)
                    for suffix in lambda_var_suffixes:
                        name = (lambda_variable_name + suffix)
                        force.addGlobalParameter(name, 1.0)
                else:  # fix lambda variable to 1.0
                    for suffix in lambda_var_suffixes:
                        name = (lambda_variable_name + suffix)
                        energy_expression = energy_expression + name + '=1.0;'
                    force = force_cls(energy_expression)
                return force

            # Create CustomNonbondedForces to handle sterics particle interactions between
            # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
            # to 1.0 for decoupled interactions in alchemical/alchemical force.
            if len(lambda_var_suffixes) > 1:
                aa_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                                 'lambda_sterics', lambda_var_suffixes, is_lambda_controlled=True)
                all_sterics_custom_nonbonded_forces = [aa_sterics_custom_nonbonded_force]
            else:
                na_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                                 'lambda_sterics', lambda_var_suffixes, is_lambda_controlled=True)
                aa_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                                 'lambda_sterics', lambda_var_suffixes, alchemical_region.annihilate_sterics)
                all_sterics_custom_nonbonded_forces = [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]

            # Add parameters and configure CustomNonbondedForces to match reference force.
            for force in all_sterics_custom_nonbonded_forces:
                force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
                force.addPerParticleParameter("epsilon")  # Lennard-Jones epsilon
                force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
                force.setCutoffDistance(nonbonded_force.getCutoffDistance())
                force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
                if self.disable_alchemical_dispersion_correction:
                    force.setUseLongRangeCorrection(False)
                else:
                    force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())

                if is_periodic_method:
                    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
                else:
                    force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

            if use_exact_pme_treatment:
                #electrostatics are handled by offset
                all_electrostatics_custom_nonbonded_forces = []
            else:
                # Create CustomNonbondedForces to handle electrostatics particle interactions between
                # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
                # to 1.0 for decoupled interactions in alchemical/alchemical force.
                if len(lambda_var_suffixes) > 1:
                    aa_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                                            'lambda_electrostatics', lambda_var_suffixes, is_lambda_controlled=True)
                    all_electrostatics_custom_nonbonded_forces = [aa_electrostatics_custom_nonbonded_force]
                else:
                    na_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                                            'lambda_electrostatics', lambda_var_suffixes, is_lambda_controlled=True)
                    aa_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                                            'lambda_electrostatics', lambda_var_suffixes,  alchemical_region.annihilate_electrostatics)
                    all_electrostatics_custom_nonbonded_forces = [na_electrostatics_custom_nonbonded_force,
                                                                  aa_electrostatics_custom_nonbonded_force]

            # Common parameters and configuration for electrostatics CustomNonbondedForces.
            for force in all_electrostatics_custom_nonbonded_forces:
                force.addPerParticleParameter("charge")  # partial charge
                force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
                if ((is_ewald_method and self.alchemical_pme_treatment == 'coulomb') or
                        (is_rf_method and self.alchemical_rf_treatment == 'switched')):
                    # Use switching function for alchemical electrostatics to ensure force continuity at cutoff.
                    force.setUseSwitchingFunction(True)
                else:
                    force.setUseSwitchingFunction(False)
                force.setSwitchingDistance(nonbonded_force.getCutoffDistance() - self.switch_width)
                force.setCutoffDistance(nonbonded_force.getCutoffDistance())
                force.setUseLongRangeCorrection(False)  # long-range dispersion correction is meaningless for electrostatics
                if is_periodic_method:
                    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
                else:
                    force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

            # Create CustomBondForces to handle sterics 1,4 exceptions interactions between
            # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
            # to 1.0 for decoupled interactions in alchemical/alchemical force.
            if len(lambda_var_suffixes) > 1:
                aa_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                            'lambda_sterics', lambda_var_suffixes, is_lambda_controlled=True)
                all_sterics_custom_bond_forces = [aa_sterics_custom_bond_force]
            else:
                na_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                            'lambda_sterics', lambda_var_suffixes, is_lambda_controlled=True)
                aa_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                            'lambda_sterics', lambda_var_suffixes, alchemical_region.annihilate_sterics)
                all_sterics_custom_bond_forces = [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]

            for force in all_sterics_custom_bond_forces:
                force.addPerBondParameter("sigma")  # Lennard-Jones effective sigma
                force.addPerBondParameter("epsilon")  # Lennard-Jones effective epsilon

            # With exact PME treatment, exception electrostatics is handled through offset parameters.
            if use_exact_pme_treatment:
                all_electrostatics_custom_bond_forces = []
            else:
                # Create CustomBondForces to handle electrostatics 1,4 exceptions interactions between
                # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
                # to 1.0 for decoupled interactions in alchemical/alchemical force.
                if len(lambda_var_suffixes) > 1:
                    aa_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                                       'lambda_electrostatics', lambda_var_suffixes, is_lambda_controlled=True)
                    all_electrostatics_custom_bond_forces = [aa_electrostatics_custom_bond_force]
                else:
                    na_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                                       'lambda_electrostatics', lambda_var_suffixes, is_lambda_controlled=True)
                    aa_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                                       'lambda_electrostatics', lambda_var_suffixes, alchemical_region.annihilate_electrostatics)
                    all_electrostatics_custom_bond_forces = [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]
            # Create CustomBondForce to handle exceptions for electrostatics
            for force in all_electrostatics_custom_bond_forces:
                force.addPerBondParameter("chargeprod")  # charge product
                force.addPerBondParameter("sigma")  # Lennard-Jones effective sigma

            # -------------------------------------------------------------------------------
            # Distribute particle interactions contributions in appropriate nonbonded forces
            # -------------------------------------------------------------------------------

            # Create atom groups.
            alchemical_atomsets = [region.alchemical_atoms for region in alchemical_regions_pairs]

            # Copy NonbondedForce particle terms for alchemically-modified particles
            # to CustomNonbondedForces, and/or add the charge offsets for exact PME.
            # On CUDA, for efficiency reasons, all nonbonded forces (custom and not)
            # must have the same particles.
            for particle_index in range(nonbonded_force.getNumParticles()):
                # Retrieve nonbonded parameters.
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
                # Set sterics parameters in CustomNonbondedForces.
                for force in all_sterics_custom_nonbonded_forces:
                    force.addParticle([sigma, epsilon])
                # Set electrostatics parameters in CustomNonbondedForces.
                for force in all_electrostatics_custom_nonbonded_forces:
                    force.addParticle([charge, sigma])
                # Set offset parameters in NonbondedForce.
                if use_exact_pme_treatment and particle_index in alchemical_atomsets[0] and len(lambda_var_suffixes) == 1:
                    nonbonded_force.addParticleParameterOffset('lambda_electrostatics{}'.format(lambda_var_suffixes[0]),
                                                               particle_index, charge, 0.0, 0.0)

            # Turn off interactions contribution from alchemically-modified particles in unmodified
            # NonbondedForce that will be handled by all other forces
            for particle_index in range(nonbonded_force.getNumParticles()):
                # Retrieve parameters.
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
                # Even with exact treatment of the PME electrostatics, we turn off
                # the NonbondedForce charge which is modeled by the offset parameter.
                if particle_index in alchemical_atomsets[0]:
                    nonbonded_force.setParticleParameters(particle_index, abs(0.0*charge), sigma, abs(0*epsilon))
            # Restrict interaction evaluation of CustomNonbondedForces to their respective atom groups.
            # Sterics
            if len(lambda_var_suffixes) == 1:
                logger.debug('Adding steric interaction groups between {} and the environment.'.format(lambda_var_suffixes[0]))
                na_sterics_custom_nonbonded_force.addInteractionGroup(nonalchemical_atomset, alchemical_atomsets[0])

            logger.debug('Adding a steric interaction group between group {0} and {1}.'.format(lambda_var_suffixes[0],
                                                                                               lambda_var_suffixes[-1]))
            aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atomsets[0], alchemical_atomsets[-1])

            # Electrostatics
            if not use_exact_pme_treatment:
                if len(lambda_var_suffixes) == 1:
                    logger.debug('Adding electrostatic interaction groups between {} and the environment.'.format(lambda_var_suffixes[0]))
                    na_electrostatics_custom_nonbonded_force.addInteractionGroup(nonalchemical_atomset, alchemical_atomsets[0])

                logger.debug('Adding a electrostatic interaction group between group {0} and {1}.'.format(lambda_var_suffixes[0],
                                                                                                          lambda_var_suffixes[-1]))
                aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atomsets[0], alchemical_atomsets[-1])

            else:
                # Using the nonbonded force to handle electrostatics
                # and the "interaction groups" in the nonbonded have already been handled by exclusions.
                pass

            # ---------------------------------------------------------------
            # Distribute exceptions contributions in appropriate bond forces
            # ---------------------------------------------------------------
            all_custom_nonbonded_forces = all_sterics_custom_nonbonded_forces + all_electrostatics_custom_nonbonded_forces
            # Move all NonbondedForce exception terms for alchemically-modified particles to CustomBondForces.
            for exception_index in range(nonbonded_force.getNumExceptions()):
                # Retrieve parameters.
                iatom, jatom, chargeprod, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)

                # Exclude this atom pair in CustomNonbondedForces. All nonbonded forces
                # must have the same number of exceptions/exclusions on CUDA platform.
                for force in all_custom_nonbonded_forces:
                    force.addExclusion(iatom, jatom)

                # Check if this is an exception or an exclusion
                is_exception_epsilon = abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 0.0
                is_exception_chargeprod = abs(chargeprod.value_in_unit_system(unit.md_unit_system)) > 0.0

                # Check how many alchemical atoms we have in the exception.
                if len(lambda_var_suffixes) > 1:
                    # Pair of interacting alchemical regions, therefore they are both alchemical or neither alchemical.
                    both_alchemical = ((iatom in alchemical_atomsets[0] and jatom in alchemical_atomsets[1]) or
                                       (jatom in alchemical_atomsets[0] and iatom in alchemical_atomsets[1]))
                    only_one_alchemical = False
                    #The condition of at_least_one_alchemical is treated only once per single
                    # region so we don't repeat it when dealing with pairs of interacting regions.
                    at_least_one_alchemical = False

                    if use_exact_pme_treatment and both_alchemical and is_exception_chargeprod:
                        # Exceptions here should be scaled by lam0*lam1.
                        # This can be implemented in the future using a CustomBondForce.
                        raise ValueError('Cannot have exception that straddles two alchemical regions')

                else:
                    # Single alchemical region.
                    both_alchemical = iatom in alchemical_atomsets[0] and jatom in alchemical_atomsets[0]
                    at_least_one_alchemical = iatom in alchemical_atomsets[0] or jatom in alchemical_atomsets[0]
                    only_one_alchemical = at_least_one_alchemical and not both_alchemical

                    # If this is an electrostatic exception and we're using exact PME,
                    # we just have to add the exception offset to the NonbondedForce.
                    if use_exact_pme_treatment and at_least_one_alchemical and is_exception_chargeprod:
                        nonbonded_force.addExceptionParameterOffset('lambda_electrostatics{}'.format(lambda_var_suffixes[0]),
                                                                    exception_index, chargeprod, 0.0, 0.0)

                # If exception (and not exclusion), add special CustomBondForce terms to
                # handle alchemically-modified Lennard-Jones and electrostatics exceptions
                if both_alchemical:
                    if is_exception_epsilon:
                        aa_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                    if is_exception_chargeprod and not use_exact_pme_treatment:
                        aa_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma])

                # When this is a single region we model the exception between alchemical
                # and non-alchemical particles using a single custom bond.
                elif only_one_alchemical:
                    if is_exception_epsilon:
                        na_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                    if is_exception_chargeprod and not use_exact_pme_treatment:
                        na_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma])
                # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce

                # Turn off all exception contributions from alchemical atoms in the NonbondedForce
                # modelling non-alchemical atoms only. We need to do it only once per single
                # region so we don't repeat it when dealing with pairs of interacting regions.
                if at_least_one_alchemical:
                    nonbonded_force.setExceptionParameters(exception_index, iatom, jatom,
                                                           abs(0.0*chargeprod), sigma, abs(0.0*epsilon))
            # Add global parameters to forces.
            def add_global_parameters(force):
                force.addGlobalParameter('softcore_alpha', alchemical_region.softcore_alpha)
                force.addGlobalParameter('softcore_beta', alchemical_region.softcore_beta)
                force.addGlobalParameter('softcore_a', alchemical_region.softcore_a)
                force.addGlobalParameter('softcore_b', alchemical_region.softcore_b)
                force.addGlobalParameter('softcore_c', alchemical_region.softcore_c)
                force.addGlobalParameter('softcore_d', alchemical_region.softcore_d)
                force.addGlobalParameter('softcore_e', alchemical_region.softcore_e)
                force.addGlobalParameter('softcore_f', alchemical_region.softcore_f)

            all_custom_forces = (all_custom_nonbonded_forces +
                                 all_sterics_custom_bond_forces +
                                 all_electrostatics_custom_bond_forces)
            for force in all_custom_forces:
                add_global_parameters(force)
            # With exact treatment of PME electrostatics, the NonbondedForce
            # is affected by lambda electrostatics as well.
            if 'lambda_sterics{}'.format(lambda_var_suffixes[0]) in forces_by_lambda:
                forces_by_lambda['lambda_electrostatics{}'.format(lambda_var_suffixes[0])].extend(all_electrostatics_custom_nonbonded_forces + all_electrostatics_custom_bond_forces)
                forces_by_lambda['lambda_sterics{}'.format(lambda_var_suffixes[0])].extend(all_sterics_custom_nonbonded_forces + all_sterics_custom_bond_forces)
            else:
                forces_by_lambda['lambda_electrostatics{}'.format(lambda_var_suffixes[0])] = all_electrostatics_custom_nonbonded_forces + all_electrostatics_custom_bond_forces
                forces_by_lambda['lambda_sterics{}'.format(lambda_var_suffixes[0])] = all_sterics_custom_nonbonded_forces + all_sterics_custom_bond_forces
        # CAVE: I modified this, pushed this into the inner loop, was one level up
        if use_exact_pme_treatment:
            forces_by_lambda['lambda_electrostatics{}'.format(lambda_var_suffixes[0])].append(nonbonded_force)
        else:
            forces_by_lambda[''] = [nonbonded_force]
        return forces_by_lambda

    def _alchemically_modify_CustomNonbondedForce(self, reference_force, _, __):
        """NYI: Create alchemically-modified version of a CustomNonbondedForce generated from the LennardJonesGenerator.

        When working, this will correctly modify the CustomNonbondedForce made by the OpenMM forcefield.py file's
        "LennardJonesGenerator" function which creates tabulated potentials for the C6 and C12 interactions
        (i.e. Acoef = 4es^12 and Bcoef = 4es^6)

        However, this has not been implemented as there are some technical problems which need to be resolved
        both in this code (flow of converting "native" nonbonded interactions from multiple forces (NB + CustomNB)
        into alchemical forces) and in the science (How do you softcore potential a "A/r^12 - B/r^6" form where A and B
        are tabulated?).


        See https://github.com/choderalab/openmmtools/issues/510 for more details on this problem.

        """
        # See https://github.com/openmm/openmm/blob/be19e0222ddf66f612016a3c1f687161a53c2396/wrappers/python/openmm/app/forcefield.py#L2548-L2572
        lj_generator_expression = 'acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6'
        reference_energy_expression = reference_force.getEnergyFunction()
        if lj_generator_expression in reference_energy_expression:
            error_msg = (f"Unsupported Native Lennard Jones representation!\n"
                         f"There is a CustomNonbondedForce which contains the energy function:\n"
                         f"\t{lj_generator_expression}\n"
                         f"Which is likely from the OpenMM forcefield.py file for Tabulated Lennard Jones functions"
                         f"such as the CHARMM36 FF. Supporting this type of LJ representation is not yet implemented.\n"
                         f"See https://github.com/choderalab/openmmtools/issues/510 for more details."
                         )
            raise NotImplementedError(error_msg)
        # Don't create a force if you made it here (remove when implemented)
        return {'': [copy.deepcopy(reference_force)]}

    def _alchemically_modify_AmoebaMultipoleForce(self, reference_force, alchemical_region, _):
        if len(alchemical_region) > 1:
            raise NotImplementedError("Multiple regions does not work with AmoebaMultipoleForce")
        else:
            alchemical_region = alchemical_region[0]
        raise Exception("Not implemented; needs CustomMultipleForce")

    def _alchemically_modify_AmoebaVdwForce(self, reference_force, alchemical_region, _):
        """Create alchemically-modified version of AmoebaVdwForce.

        Not implemented.

        TODO
        ----
        * Supported periodic boundary conditions need to be handled correctly.
        * Exceptions/exclusions need to be dealt with.

        """
        # This feature is incompletely implemented, so raise an exception.
        if len(alchemical_region) > 1:
            raise NotImplementedError("Multiple regions does not work with AmoebaVdwForce")
        else:
            alchemical_region = alchemical_region[0]
        raise NotImplementedError('Alchemical modification of Amoeba VdW Forces is not supported.')

        # Softcore Halgren potential from Eq. 3 of
        # Shi, Y., Jiao, D., Schnieders, M.J., and Ren, P. (2009). Trypsin-ligand binding free
        # energy calculation with AMOEBA. Conf Proc IEEE Eng Med Biol Soc 2009, 2328-2331.
        energy_expression = 'lambda^5 * epsilon * (1.07^7 / (0.7*(1-lambda)^2+(rho+0.07)^7)) * (1.12 / (0.7*(1-lambda)^2 + rho^7 + 0.12) - 2);'
        energy_expression += 'epsilon = 4*epsilon1*epsilon2 / (sqrt(epsilon1) + sqrt(epsilon2))^2;'
        energy_expression += 'rho = r / R0;'
        energy_expression += 'R0 = (R01^3 + R02^3) / (R01^2 + R02^2);'
        energy_expression += 'lambda = vdw_lambda * (ligand1*(1-ligand2) + ligand2*(1-ligand1)) + ligand1*ligand2;'
        energy_expression += 'vdw_lambda = %f;' % vdw_lambda

        softcore_force = openmm.CustomNonbondedForce(energy_expression)
        softcore_force.addPerParticleParameter('epsilon')
        softcore_force.addPerParticleParameter('R0')
        softcore_force.addPerParticleParameter('ligand')

        for particle_index in range(system.getNumParticles()):
            # Retrieve parameters from vdW force.
            [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
            # Add parameters to CustomNonbondedForce.
            if particle_index in alchemical_region.alchemical_atoms:
                softcore_force.addParticle([epsilon, sigma, 1])
            else:
                softcore_force.addParticle([epsilon, sigma, 0])

            # Deal with exclusions.
            excluded_atoms = force.getParticleExclusions(particle_index)
            for jatom in excluded_atoms:
                if (particle_index < jatom):
                    softcore_force.addExclusion(particle_index, jatom)

        # Make sure periodic boundary conditions are treated the same way.
        # TODO: Handle PBC correctly.
        softcore_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
        softcore_force.setCutoffDistance( force.getCutoff() )

        # Turn off vdW interactions for alchemically-modified atoms.
        for particle_index in ligand_atoms:
            # Retrieve parameters.
            [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
            epsilon = 1.0e-6 * epsilon # TODO: For some reason, we cannot set epsilon to 0.
            force.setParticleParameters(particle_index, parentIndex, sigma, epsilon, reductionFactor)

        # Deal with exceptions here.
        # TODO

        return [softcore_force]

    @staticmethod
    def _alchemically_modify_GBSAOBCForce(reference_force, alchemical_region, _, sasa_model='ACE'):
        """Create alchemically-modified version of GBSAOBCForce.

        Parameters
        ----------
        reference_force : openmm.GBSAOBCForce
            The reference GBSAOBCForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the atoms to
            alchemically modify.
        sasa_model : str, optional, default='ACE'
            Solvent accessible surface area model.

        Returns
        -------
        custom_force : openmm.CustomGBForce
            The alchemical version of the reference force.

        TODO
        ----
        * Add support for all types of GBSA forces supported by OpenMM.
        * Can we more generally modify any CustomGBSAForce?

        """
        if len(alchemical_region) > 1:
            raise NotImplementedError("Multiple regions does not work with GBSAOBCForce")
        else:
            alchemical_region = alchemical_region[0]

        # TODO make sasa_model a Factory attribute?
        custom_force = openmm.CustomGBForce()

        # Add per-particle parameters.
        custom_force.addGlobalParameter("lambda_electrostatics", 1.0)
        custom_force.addPerParticleParameter("charge")
        custom_force.addPerParticleParameter("radius")
        custom_force.addPerParticleParameter("scale")
        custom_force.addPerParticleParameter("alchemical")

        # Set nonbonded method.
        custom_force.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom_force.setCutoffDistance(reference_force.getCutoffDistance())

        # Add global parameters.
        custom_force.addGlobalParameter("solventDielectric", reference_force.getSolventDielectric())
        custom_force.addGlobalParameter("soluteDielectric", reference_force.getSoluteDielectric())
        custom_force.addGlobalParameter("offset", 0.009)

        custom_force.addComputedValue("I",  "(lambda_electrostatics*alchemical2 + (1-alchemical2))*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r+C);"
                                "U=r+sr2;"
                                "C=2*(1/or1-1/L)*step(sr2-r-or1);"
                                "L=max(or1, D);"
                                "D=abs(r-sr2);"
                                "sr2 = scale2*or2;"
                                "or1 = radius1-offset; or2 = radius2-offset", openmm.CustomGBForce.ParticlePairNoExclusions)

        custom_force.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", openmm.CustomGBForce.SingleParticle)

        custom_force.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*(lambda_electrostatics*alchemical+(1-alchemical))*charge^2/B", openmm.CustomGBForce.SingleParticle)
        if sasa_model == 'ACE':
            custom_force.addEnergyTerm("(lambda_electrostatics*alchemical+(1-alchemical))*28.3919551*(radius+0.14)^2*(radius/B)^6", openmm.CustomGBForce.SingleParticle)

        custom_force.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*(lambda_electrostatics*alchemical1+(1-alchemical1))*charge1*(lambda_electrostatics*alchemical2+(1-alchemical2))*charge2/f;"
                             "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", openmm.CustomGBForce.ParticlePairNoExclusions);

        # Add particle parameters.
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
            # Set particle parameters.
            if particle_index in alchemical_region.alchemical_atoms:
                parameters = [charge, radius, scaling_factor, 1.0]
            else:
                parameters = [charge, radius, scaling_factor, 0.0]
            custom_force.addParticle(parameters)

        return {'lambda_electrostatics': [custom_force]}

    def _alchemically_modify_CustomGBForce(self, reference_force, alchemical_region, _):
        """Create alchemically-modified version of CustomGBForce.

        The GB functions are meta-programmed using the following rules:
        - 'lambda_electrostatics' is added as a global parameter.
        - 'alchemical' is added as a per-particle parameter. All atoms in
          the alchemical group have this parameter set to 1; otherwise 0.
        - Any single-particle energy term (`CustomGBForce.SingleParticle`)
          is scaled by `(lambda_electrostatics*alchemical+(1-alchemical))`
        - Any two-particle energy term (`CustomGBForce.ParticlePairNoExclusions`)
          has charge 1 (`charge1`) replaced by `(lambda_electrostatics*alchemical1+(1-alchemical1))*charge1`
          and charge 2 (`charge2`) replaced by `(lambda_electrostatics*alchemical2+(1-alchemical2))*charge2`.
        - Any single-particle computed value (`CustomGBForce.SingleParticle`)
          remains unmodified
        - Any two-particle computed value (`CustomGBForce.ParticlePairNoExclusions`)
          is scaled by `(lambda_electrostatics*alchemical2 + (1-alchemical2))`

        Scaling of a term should always prepend and capture the value with
        an intermediate variable. For example, prepending `scaling * unscaled; unscaled =`
        will capture the value of the expression as `unscaled` and multiple by `scaled`.
        This avoids the need to identify the head expression and add parentheses.

        .. warning::
            This may not work correctly for all GB models.

        Parameters
        ----------
        reference_force : openmm.GBSAOBCForce
            The reference GBSAOBCForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the atoms to
            alchemically modify.

        Returns
        -------
        custom_force : openmm.CustomGBForce
            The alchemical version of the reference force.

        """
        if len(alchemical_region) > 1:
            raise NotImplementedError("Multiple regions does not work with CustomGBForce")
        else:
            alchemical_region = alchemical_region[0]

        custom_force = openmm.CustomGBForce()

        # Add global parameters
        for index in range(reference_force.getNumGlobalParameters()):
            name = reference_force.getGlobalParameterName(index)
            default_value = reference_force.getGlobalParameterDefaultValue(index)
            custom_force.addGlobalParameter(name, default_value)
        custom_force.addGlobalParameter("lambda_electrostatics", 1.0)

        # Add per-particle parameters.
        for index in range(reference_force.getNumPerParticleParameters()):
            name = reference_force.getPerParticleParameterName(index)
            custom_force.addPerParticleParameter(name)
        custom_force.addPerParticleParameter("alchemical")

        # Set nonbonded methods.
        custom_force.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom_force.setCutoffDistance(reference_force.getCutoffDistance())

        # Add computations.
        for index in range(reference_force.getNumComputedValues()):
            name, expression, computation_type = reference_force.getComputedValueParameters(index)

            # Alter expression for particle pair terms only.
            if computation_type is not openmm.CustomGBForce.SingleParticle:
                prepend = ('alchemical_scaling*unscaled; '
                           'alchemical_scaling = (lambda_electrostatics*alchemical2 + (1-alchemical2)); '
                           'unscaled = ')
                expression = prepend + expression

            custom_force.addComputedValue(name, expression, computation_type)

        # Add energy terms.
        for index in range(reference_force.getNumEnergyTerms()):
            expression, computation_type = reference_force.getEnergyTermParameters(index)

            # Alter expressions
            if computation_type is openmm.CustomGBForce.SingleParticle:
                prepend = ('alchemical_scaling*unscaled; '
                           'alchemical_scaling = (lambda_electrostatics*alchemical + (1-alchemical)); '
                           'unscaled = ')
                expression = prepend + expression
            else:
                expression = expression.replace('charge1', 'alchemically_scaled_charge1')
                expression = expression.replace('charge2', 'alchemically_scaled_charge2')
                expression += ' ; alchemically_scaled_charge1 = (lambda_electrostatics*alchemical1+(1-alchemical1)) * charge1;'
                expression += ' ; alchemically_scaled_charge2 = (lambda_electrostatics*alchemical2+(1-alchemical2)) * charge2;'

            custom_force.addEnergyTerm(expression, computation_type)

        # Add particle parameters
        for particle_index in range(reference_force.getNumParticles()):
            parameters = reference_force.getParticleParameters(particle_index)
            # Append alchemical parameter
            parameters = list(parameters)
            if particle_index in alchemical_region.alchemical_atoms:
                parameters.append(1.0)
            else:
                parameters.append(0.0)
            custom_force.addParticle(parameters)

        # Add tabulated functions
        for function_index in range(reference_force.getNumTabulatedFunctions()):
            name = reference_force.getTabulatedFunctionName(function_index)
            function = reference_force.getTabulatedFunction(function_index)
            function_copy = copy.deepcopy(function)
            custom_force.addTabulatedFunction(name, function_copy)

        # Add exclusions
        for exclusion_index in range(reference_force.getNumExclusions()):
            [particle1, particle2] = reference_force.getExclusionParticles(exclusion_index)
            custom_force.addExclusion(particle1, particle2)

        return {'lambda_electrostatics': [custom_force]}

    # -------------------------------------------------------------------------
    # Internal usage: Infer force labels
    # -------------------------------------------------------------------------

    @staticmethod
    def _find_force_components(alchemical_system):
        """Return force labels and indices for each force."""
        def add_label(label, index):
            assert label not in force_labels
            force_labels[label] = index

        def check_parameter(custom_force, parameter):
            for parameter_id in range(custom_force.getNumGlobalParameters()):
                parameter_name = custom_force.getGlobalParameterName(parameter_id)
                if parameter_name.startswith(parameter):
                    region_name = parameter_name[len(parameter)+1:] if len(parameter_name) > len(parameter) else None
                    return [True, region_name]
            return [False, None]

        def check_energy_expression(custom_force, parameter):
            try:
                energy_expression = custom_force.getEnergyFunction()
            except AttributeError:  # CustomGBForce
                found = False
                for index in range(custom_force.getNumEnergyTerms()):
                    expression, _ = custom_force.getEnergyTermParameters(index)
                    if parameter in expression:
                        found = True
                        break
                return [found, ''] # CustomGBForce will not have a parameter suffix
            found = parameter in energy_expression
            if not found:
                return [found, ''] # param not found and by extension no parameter suffix
            #Look for parameter suffix
            p = re.compile(r'{}_([A-Za-z0-9]+)'.format(parameter))
            try:
                suffix = p.search(energy_expression).group(1)
                return [found, suffix]
            except AttributeError:
                return [found, ''] # no parameter suffix found

        force_labels = {}
        nonbonded_forces = {}
        sterics_bond_forces = {}
        electro_bond_forces = {}

        # We save CustomBondForces and CustomNonbondedForces used for nonbonded
        # forces and exceptions to distinguish them later
        for force_index, force in enumerate(alchemical_system.getForces()):
            if isinstance(force, openmm.CustomAngleForce):
                parameter_found, region_name = check_parameter(force, 'lambda_angles')
                if parameter_found:
                    label = 'alchemically modified HarmonicAngleForce'
                    if region_name is not None:
                        label = label + ' for region ' + region_name
                    add_label(label, force_index)

            elif isinstance(force, openmm.CustomBondForce) and check_parameter(force, 'lambda_bonds')[0]:
                #Need extra check for 'lambda_bonds' in the if to differntiate from exceptions
                parameter_found, region_name = check_parameter(force, 'lambda_bonds')
                if parameter_found:
                    label = 'alchemically modified HarmonicBondForce'
                    if region_name is not None:
                        label = label + ' for region ' + region_name
                    add_label(label, force_index)

            elif isinstance(force, openmm.CustomTorsionForce):
                parameter_found, region_name = check_parameter(force, 'lambda_torsions')
                if parameter_found:
                    label = 'alchemically modified PeriodicTorsionForce'
                    if region_name is not None:
                        label = label + ' for region ' + region_name
                    add_label(label, force_index)

            elif isinstance(force, openmm.CustomGBForce):
                parameter_found, _ = check_parameter(force, 'lambda_electrostatics')
                if parameter_found:
                    #No multi region for GBForce
                    if check_energy_expression(force, 'unscaled')[0]:
                        add_label('alchemically modified CustomGBForce', force_index)
                    else:
                        add_label('alchemically modified GBSAOBCForce', force_index)

            elif isinstance(force, openmm.CustomBondForce):
                parameter_found, region_type_suffix = check_energy_expression(force, 'lambda')
                if parameter_found:
                    _, region_name_suffix = check_energy_expression(force, 'lambda_{}'.format(region_type_suffix))
                    if region_type_suffix == 'sterics':
                        if region_name_suffix in sterics_bond_forces:
                            sterics_bond_forces[region_name_suffix].append([force_index, force])
                        else:
                            sterics_bond_forces[region_name_suffix] = [[force_index, force]]
                    if region_type_suffix == 'electrostatics':
                        if region_name_suffix in electro_bond_forces:
                            electro_bond_forces[region_name_suffix].append([force_index, force])
                        else:
                            electro_bond_forces[region_name_suffix] = [[force_index, force]]

            elif (isinstance(force, openmm.CustomNonbondedForce) and force.getEnergyFunction() == '0.0;' and
                          force.getGlobalParameterName(0) == 'lambda_electrostatics'):
                add_label('CustomNonbondedForce holding alchemical atoms unmodified charges', force_index)

            elif isinstance(force, openmm.CustomNonbondedForce):
                parameter_found, region_type_suffix = check_energy_expression(force, 'lambda')
                if parameter_found:
                    _, region_name_suffix = check_energy_expression(force, 'lambda_{}'.format(region_type_suffix))
                    if region_type_suffix == 'sterics':
                        if region_name_suffix in nonbonded_forces:
                            nonbonded_forces[region_name_suffix].append(['sterics', force_index, force])
                        else:
                            nonbonded_forces[region_name_suffix] = [['sterics', force_index, force]]
                    if region_type_suffix == 'electrostatics':
                        if region_name_suffix in nonbonded_forces:
                            nonbonded_forces[region_name_suffix].append(['electrostatics', force_index, force])
                        else:
                            nonbonded_forces[region_name_suffix] = [['electrostatics', force_index, force]]
            else:
                add_label('unmodified ' + force.__class__.__name__, force_index)

        # Differentiate between na/aa nonbonded forces.
        alchemical_atoms_by_region = {}
        for region_name, forces in nonbonded_forces.items():
            for force_type, force_index, force in forces:
                if region_name == '':
                    label = 'alchemically modified NonbondedForce for {}alchemical/alchemical ' + force_type
                else:
                    label = 'alchemically modified NonbondedForce for {}alchemical/alchemical ' + force_type +\
                            ' for region ' + region_name
                interacting_atoms, alchemical_atoms = force.getInteractionGroupParameters(0)
                alchemical_atoms_by_region[region_name] = [interacting_atoms, alchemical_atoms]
                if interacting_atoms == alchemical_atoms:  # alchemical-alchemical atoms
                    add_label(label.format(''), force_index)
                else:
                    add_label(label.format('non-'), force_index)

        # Initialize sterics_bond_forces or electro_bond_forces for regions which did not have forces, as empty lists
        for k in sterics_bond_forces.keys():
            if k in electro_bond_forces:
                pass
            else:
                electro_bond_forces[k] = []
        for k in electro_bond_forces.keys():
            if k in sterics_bond_forces:
                pass
            else:
                sterics_bond_forces[k] = []

        # Differentiate between na/aa bond forces for exceptions.
        for (region_name, sterics_forces), electro_forces in zip(sterics_bond_forces.items(), electro_bond_forces.values()):
            if len(sterics_forces) == 0:
                continue
            for force_type, bond_forces in [('sterics', sterics_forces), ('electrostatics', electro_forces)]:
                # With exact PME there are no CustomBondForces modeling electrostatics exceptions.
                if force_type == 'electrostatics' and len(bond_forces) == 0:
                    continue
                # Otherwise there should be two CustomBondForce.
                assert len(bond_forces) == 2
                if region_name == '':
                    label = 'alchemically modified BondForce for {}alchemical/alchemical ' + force_type + ' exceptions'
                else:
                    label = 'alchemically modified BondForce for {}alchemical/alchemical ' + force_type +\
                            ' exceptions for region ' + region_name

                # Sort forces by number of bonds.
                bond_forces = sorted(bond_forces, key=lambda x: x[1].getNumBonds())
                (force_index1, force1), (force_index2, force2) = bond_forces

                # Check if both define their parameters (with decoupling the lambda
                # parameter doesn't exist in the alchemical-alchemical force)
                parameter_name = 'lambda_' + force_type
                if region_name != '':
                    parameter_name = parameter_name + '_{}'.format(region_name)
                interacting_atoms, alchemical_atoms = alchemical_atoms_by_region[region_name]

                if check_parameter(force1, parameter_name)[0] != check_parameter(force2, parameter_name)[0]:
                    if check_parameter(force1, parameter_name)[0]:
                        add_label(label.format('non-'), force_index1)
                        add_label(label.format(''), force_index2)
                    else:
                        add_label(label.format(''), force_index1)
                        add_label(label.format('non-'), force_index2)

                # If they are both empty they are identical and any label works.
                elif force1.getNumBonds() == 0 and force2.getNumBonds() == 0:
                    add_label(label.format(''), force_index1)
                    add_label(label.format('non-'), force_index2)

                # We check that the bond atoms are both alchemical or not.
                else:
                    atom_i, atom_j, _ = force2.getBondParameters(0)
                    both_alchemical = atom_i in alchemical_atoms and atom_j in alchemical_atoms
                    if both_alchemical:
                        add_label(label.format(''), force_index2)
                        add_label(label.format('non-'), force_index1)
                    else:
                        add_label(label.format('non-'), force_index2)
                        add_label(label.format(''), force_index1)
        return force_labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(AlchemicalFunction, globals())