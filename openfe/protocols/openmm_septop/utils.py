import os
import pathlib
from openmmtools import states
from openmmtools.alchemy import (
    AlchemicalStateError, AlchemicalRegion,
    AlchemicalFunction, AbsoluteAlchemicalFactory,
)
from openmmtools.states import GlobalParameterState

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


class SepTopParameterState(GlobalParameterState):
    """
    Composable state to control lambda parameters for two ligands.
    See :class:`openmmtools.states.GlobalParameterState` for more details.
    Parameters
    ----------
    parameters_name_suffix : Optional[str]
      If specified, the state will control a modified version of the parameter
      ``lambda_restraints_{parameters_name_suffix}` instead of just
      ``lambda_restraints``.
    lambda_sterics_A : Optional[float]
      The value for the vdW interactions for ligand A.
      If defined, must be between 0 and 1.
    lambda_electrosterics_A : Optional[float]
      The value for the electrostatics interactions for ligand A.
      If defined, must be between 0 and 1.
    lambda_restraints_A : Optional[float]
      The strength of the restraint for ligand A.
      If defined, must be between 0 and 1.
    lambda_bonds_A : Optional[float]
      The value for modifying bonds for ligand A.
      If defined, must be between 0 and 1.
    lambda_angles_A : Optional[float]
      The value for modifying angles for ligand A.
      If defined, must be between 0 and 1.
    lambda_dihedrals_A : Optional[float]
      The value for modifying dihedrals for ligand A.
      If defined, must be between 0 and 1.
    lambda_sterics_B : Optional[float]
      The value for the vdW interactions for ligand B.
      If defined, must be between 0 and 1.
    lambda_electrosterics_B : Optional[float]
      The value for the electrostatics interactions for ligand B.
      If defined, must be between 0 and 1.
    lambda_restraints_B : Optional[float]
      The strength of the restraint for ligand B.
      If defined, must be between 0 and 1.
    lambda_bonds_B : Optional[float]
      The value for modifying bonds for ligand B.
      If defined, must be between 0 and 1.
    lambda_angles_B : Optional[float]
      The value for modifying angles for ligand B.
      If defined, must be between 0 and 1.
    lambda_dihedrals_B : Optional[float]
      The value for modifying dihedrals for ligand B.
      If defined, must be between 0 and 1.
    """

    class _LambdaParameter(states.GlobalParameterState.GlobalParameter):
        """A global parameter in the interval [0, 1] with standard
        value 1."""

        def __init__(self, parameter_name):
            super().__init__(parameter_name, standard_value=1.0,
                             validator=self.lambda_validator)

        @staticmethod
        def lambda_validator(self, instance, parameter_value):
            if parameter_value is None:
                return parameter_value
            if not (0.0 <= parameter_value <= 1.0):
                raise ValueError('{} must be between 0 and 1.'.format(
                self.parameter_name))
            return float(parameter_value)

    # Lambda parameters for ligand A
    lambda_sterics_A = _LambdaParameter('lambda_sterics_A')
    lambda_electrostatics_A = _LambdaParameter('lambda_electrostatics_A')
    lambda_restraints_A = _LambdaParameter('lambda_restraints_A')
    lambda_bonds_A = _LambdaParameter('lambda_bonds_A')
    lambda_angles_A = _LambdaParameter('lambda_angles_A')
    lambda_torsions_A = _LambdaParameter('lambda_torsions_A')

    # Lambda parameters for ligand B
    lambda_sterics_B = _LambdaParameter('lambda_sterics_B')
    lambda_electrostatics_B = _LambdaParameter('lambda_electrostatics_B')
    lambda_restraints_B = _LambdaParameter('lambda_restraints_B')
    lambda_bonds_B = _LambdaParameter('lambda_bonds_B')
    lambda_angles_B = _LambdaParameter('lambda_angles_B')
    lambda_torsions_B = _LambdaParameter('lambda_torsions_B')
