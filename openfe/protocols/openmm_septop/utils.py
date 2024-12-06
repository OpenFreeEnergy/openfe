import os
import pathlib
from openmmtools import states
from openmmtools.alchemy import (
    AlchemicalStateError, AlchemicalRegion,
    AlchemicalFunction, AbsoluteAlchemicalFactory,
)

def serialize(item, filename: pathlib.Path):
    """
    Serialize an OpenMM System, State, or Integrator.

    Parameters
    ----------
    item : System, State, or Integrator
        The thing to be serialized
    filename : str
        The filename to serialize to
    """
    from openmm import XmlSerializer

    # Create parent directory if it doesn't exist
    filename_basedir = filename.parent
    if not filename_basedir.exists():
        os.makedirs(filename_basedir)

    if filename.suffix == ".gz":
        import gzip

        with gzip.open(filename, mode="wb") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing.encode())
    if filename.suffix == ".bz2":
        import bz2

        with bz2.open(filename, mode="wb") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing.encode())
    else:
        with open(filename, mode="w") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing)


def deserialize(filename: pathlib.Path):
    """
    Deserialize an OpenMM System, State, or Integrator.

    Parameters
    ----------
    item : System, State, or Integrator
        The thing to be serialized
    filename : str
        The filename to serialize to
    """
    from openmm import XmlSerializer

    # Create parent directory if it doesn't exist
    filename_basedir = filename.parent
    if not filename_basedir.exists():
        os.makedirs(filename_basedir)

    if filename.suffix == ".gz":
        import gzip

        with gzip.open(filename, mode="rb") as infile:
            serialized_thing = infile.read().decode()
            item = XmlSerializer.deserialize(serialized_thing)
    if filename.suffix == ".bz2":
        import bz2

        with bz2.open(filename, mode="rb") as infile:
            serialized_thing = infile.read().decode()
            item = XmlSerializer.deserialize(serialized_thing)
    else:
        with open(filename) as infile:
            serialized_thing = infile.read()
            item = XmlSerializer.deserialize(serialized_thing)

    return item


# class AlchemicalState(states.GlobalParameterState):
#     """Represent an alchemical state.
#
#     The alchemical parameters modify the Hamiltonian and affect the
#     computation of the energy. Alchemical parameters that have value
#     None are considered undefined, which means that applying this
#     state to System and Context that have that parameter as a global
#     variable will raise an AlchemicalStateError.
#
#     Parameters
#     ----------
#     parameters_name_suffix : str, optional
#         If specified, the state will control a modified version of the global
#         parameters with the name ``parameter_name + '_' + parameters_name_suffix``.
#         When this is the case, the normal parameters are not accessible.
#     lambda_sterics : float, optional
#         Scaling factor for ligand sterics (Lennard-Jones and Halgren)
#         interactions (default is 1.0).
#     lambda_electrostatics : float, optional
#         Scaling factor for ligand charges, intrinsic Born radii, and surface
#         area term (default is 1.0).
#     lambda_bonds : float, optional
#         Scaling factor for alchemically-softened bonds (default is 1.0).
#     lambda_angles : float, optional
#         Scaling factor for alchemically-softened angles (default is 1.0).
#     lambda_torsions : float, optional
#         Scaling factor for alchemically-softened torsions (default is 1.0).
#
#     Attributes
#     ----------
#     lambda_sterics
#     lambda_electrostatics
#     lambda_bonds
#     lambda_angles
#     lambda_torsions
#
#     Examples
#     --------
#     Create an alchemically modified system.
#
#     >>> from openmmtools import testsystems
#     >>> factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
#     >>> alanine_vacuum = testsystems.AlanineDipeptideVacuum().system
#     >>> alchemical_region = AlchemicalRegion(alchemical_atoms=range(22))
#     >>> alanine_alchemical_system = factory.create_alchemical_system(reference_system=alanine_vacuum,
#     ...                                                              alchemical_regions=alchemical_region)
#
#     Create a completely undefined alchemical state.
#
#     >>> alchemical_state = AlchemicalState()
#     >>> print(alchemical_state.lambda_sterics)
#     None
#     >>> alchemical_state.apply_to_system(alanine_alchemical_system)
#     Traceback (most recent call last):
#     ...
#     openmmtools.alchemy.AlchemicalStateError: The system parameter lambda_electrostatics is not defined in this state.
#
#     Create an AlchemicalState that matches the parameters defined in
#     the System.
#
#     >>> alchemical_state = AlchemicalState.from_system(alanine_alchemical_system)
#     >>> alchemical_state.lambda_sterics
#     1.0
#     >>> alchemical_state.lambda_electrostatics
#     1.0
#     >>> print(alchemical_state.lambda_angles)
#     None
#
#     AlchemicalState implement the IComposableState interface, so it can be
#     used with CompoundThermodynamicState. All the alchemical parameters are
#     accessible through the compound state.
#
#     >>> import openmm
#     >>> from openmm import unit
#     >>> thermodynamic_state = states.ThermodynamicState(system=alanine_alchemical_system,
#     ...                                                 temperature=300*unit.kelvin)
#     >>> compound_state = states.CompoundThermodynamicState(thermodynamic_state=thermodynamic_state,
#     ...                                                    composable_states=[alchemical_state])
#     >>> compound_state.lambda_sterics
#     1.0
#
#     You can control the parameters in the OpenMM Context in this state by
#     setting the state attributes.
#
#     >>> compound_state.lambda_sterics = 0.5
#     >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
#     >>> context = compound_state.create_context(integrator)
#     >>> context.getParameter('lambda_sterics')
#     0.5
#     >>> compound_state.lambda_sterics = 1.0
#     >>> compound_state.apply_to_context(context)
#     >>> context.getParameter('lambda_sterics')
#     1.0
#
#     You can express the alchemical parameters as a mathematical expression
#     involving alchemical variables. Here is an example for a two-stage function.
#
#     >>> compound_state.set_alchemical_variable('lambda', 1.0)
#     >>> compound_state.lambda_sterics = AlchemicalFunction('step_hm(lambda - 0.5) + 2*lambda * step_hm(0.5 - lambda)')
#     >>> compound_state.lambda_electrostatics = AlchemicalFunction('2*(lambda - 0.5) * step(lambda - 0.5)')
#     >>> for l in [0.0, 0.25, 0.5, 0.75, 1.0]:
#     ...     compound_state.set_alchemical_variable('lambda', l)
#     ...     print(compound_state.lambda_sterics)
#     0.0
#     0.5
#     1.0
#     1.0
#     1.0
#
#     """
#
#     _GLOBAL_PARAMETER_ERROR = AlchemicalStateError
#
#     # -------------------------------------------------------------------------
#     # Lambda properties
#     # -------------------------------------------------------------------------
#
#     class _LambdaParameter(states.GlobalParameterState.GlobalParameter):
#         """A global parameter in the interval [0, 1] with standard value 1."""
#
#         def __init__(self, parameter_name):
#             super().__init__(parameter_name, standard_value=1.0, validator=self.lambda_validator)
#
#         @staticmethod
#         def lambda_validator(self, instance, parameter_value):
#             if parameter_value is None:
#                 return parameter_value
#             if not (0.0 <= parameter_value <= 1.0):
#                 raise ValueError('{} must be between 0 and 1.'.format(self.parameter_name))
#             return float(parameter_value)
#
#     lambda_sterics_ligandA = _LambdaParameter('lambda_sterics_ligandA')
#     lambda_electrostatics_ligandB = _LambdaParameter(
#         'lambda_electrostatics_ligandB')
#     lambda_sterics_ligandB = _LambdaParameter('lambda_sterics_ligandB')
#     lambda_electrostatics_ligandA = _LambdaParameter(
#         'lambda_electrostatics_ligandA')
#     lambda_restraints_ligandA = _LambdaParameter('lambda_restraints_ligandA')
#     lambda_restraints_ligandB = _LambdaParameter('lambda_restraints_ligandB')
#     lambda_bonds = _LambdaParameter('lambda_bonds')
#     lambda_angles = _LambdaParameter('lambda_angles')
#     lambda_torsions = _LambdaParameter('lambda_torsions')
#
#     @classmethod
#     def from_system(cls, system, *args, **kwargs):
#         """Constructor reading the state from an alchemical system.
#
#         Parameters
#         ----------
#         system : openmm.System
#             An alchemically modified system in a defined alchemical state.
#         parameters_name_suffix : str, optional
#             If specified, the state will search for a modified
#             version of the alchemical parameters with the name
#             ``parameter_name + '_' + parameters_name_suffix``.
#
#         Returns
#         -------
#         The AlchemicalState object representing the alchemical state of
#         the system.
#
#         Raises
#         ------
#         AlchemicalStateError
#             If the same parameter has different values in the system, or
#             if the system has no lambda parameters.
#
#         """
#         # The function is redefined here only to provide more specific documentation for this method.
#         return super().from_system(system, *args, **kwargs)
#
#     def set_alchemical_parameters(self, new_value):
#         """Set all defined lambda parameters to the given value.
#
#         The undefined parameters (i.e. those being set to None) remain
#         undefined.
#
#         Parameters
#         ----------
#         new_value : float
#             The new value for all defined parameters.
#
#         """
#         for parameter_name in self._parameters:
#             if self._parameters[parameter_name] is not None:
#                 setattr(self, parameter_name, new_value)
#
#     # -------------------------------------------------------------------------
#     # Function variables
#     # -------------------------------------------------------------------------
#
#     def get_function_variable(self, variable_name):
#         """Return the value of the function variable.
#
#         Function variables are variables entering mathematical expressions
#         specified with ``AlchemicalFunction``, which can be use to enslave
#         a lambda parameter to arbitrary variables.
#
#         Parameters
#         ----------
#         variable_name : str
#             The name of the function variable.
#
#         Returns
#         -------
#         variable_value : float
#             The value of the function variable.
#
#         """
#         # The function is redefined here only to provide more specific documentation for this method.
#         return super().get_function_variable(variable_name)
#
#     def set_function_variable(self, variable_name, new_value):
#         """Set the value of the function variable.
#
#         Function variables are variables entering mathematical expressions
#         specified with ``AlchemicalFunction``, which can be use to enslave
#         a lambda parameter to arbitrary variables.
#
#         Parameters
#         ----------
#         variable_name : str
#             The name of the function variable.
#         new_value : float
#             The new value for the variable.
#
#         """
#         # The function is redefined here only to provide more specific documentation for this method.
#         super().set_function_variable(variable_name, new_value)
#
#     def get_alchemical_variable(self, variable_name):
#         """Return the value of the alchemical parameter.
#
#         .. warning:
#             This is deprecated. Use ``get_function_variable`` instead.
#
#         Parameters
#         ----------
#         variable_name : str
#             The name of the alchemical variable.
#
#         Returns
#         -------
#         variable_value : float
#             The value of the alchemical variable.
#         """
#         import warnings
#         warnings.warn('AlchemicalState.get_alchemical_variable is deprecated. '
#                       'Use AlchemicalState.get_function_variable instead.')
#         return super().get_function_variable(variable_name)
#
#     def set_alchemical_variable(self, variable_name, new_value):
#         """Set the value of the alchemical variable.
#
#         .. warning:
#             This is deprecated. Use ``set_function_variable`` instead.
#
#         Parameters
#         ----------
#         variable_name : str
#             The name of the alchemical variable.
#         new_value : float
#             The new value for the variable.
#
#         """
#         import warnings
#         warnings.warn('AlchemicalState.get_alchemical_variable is deprecated. '
#                       'Use AlchemicalState.get_function_variable instead.')
#         super().set_function_variable(variable_name, new_value)
#
#     # -------------------------------------------------------------------------
#     # IComposableState interface
#     # -------------------------------------------------------------------------
#
#     def apply_to_system(self, system):
#         """Set the alchemical state of the system to this.
#
#         Parameters
#         ----------
#         system : openmm.System
#             The system to modify.
#
#         Raises
#         ------
#         AlchemicalStateError
#             If the system does not have the required lambda global variables.
#
#         """
#         # The function is redefined here only to provide more specific documentation for this method.
#         super().apply_to_system(system)
#
#     def check_system_consistency(self, system):
#         """Check if the system is in this alchemical state.
#
#         It raises a AlchemicalStateError if the system is not consistent
#         with the alchemical state.
#
#         Parameters
#         ----------
#         system : openmm.System
#             The system to test.
#
#         Raises
#         ------
#         AlchemicalStateError
#             If the system is not consistent with this state.
#
#         """
#         # The function is redefined here only to provide more specific documentation for this method.
#         super().check_system_consistency(system)
#
#     def apply_to_context(self, context):
#         """Put the Context into this AlchemicalState.
#
#         Parameters
#         ----------
#         context : openmm.Context
#             The context to set.
#
#         Raises
#         ------
#         AlchemicalStateError
#             If the context does not have the required lambda global variables.
#
#         """
#         # The function is redefined here only to provide more specific documentation for this method.
#         super().apply_to_context(context)
