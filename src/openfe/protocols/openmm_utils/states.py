# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
# 
# Acknowledgements:
# This module derives heavily from the AlchemicalState class
# in openmmtools.alchemy (https://github.com/choderalab/openmmtools).

from openmmtools.states import GlobalParameterState


class _LambdaParameter(GlobalParameterState.GlobalParameter):
    """
    A global parameter in the interval [0,, 1] with standard value 1.
    """
    def __init__(self, parameter_name):
        super().__init__(parameter_name, standard_value=1.0, validator=self.lambda_validator)

    @staticmethod
    def lambda_validator(self, instance, parameter_value):
        if parameter_value is None:
            return parameter_value
        if not (0.0 <= parameter_value <= 1.0):
            raise ValueError('{} must be between 0 and 1.'.format(self.parameter_name))
        return float(parameter_value)


class SingleRegionAlchemicalState(GlobalParameterState):
    """
    Composable state to control lambda parameters for a single
    alchemical molecule / region (``ligand_A``).

    Parameters
    ----------
    parameters_name_suffix : str | None
      If specified, the state will control a modified version of the parameter
      ``lambda_restraints_{parameters_name_suffix}` instead of just
      ``lambda_restraints``.
    lambda_sterics_A : float | None
      Control parameter for the vdW interactions for ligand A.
      If defined, must be between 0 and 1.
    lambda_electrostatics_A : float | None
      Control parameter for the electrostatics interactions for ligand A.
      If defined, must be between 0 and 1.
    lambda_bonds_A : float | None
      Control parameter for alchemically modified bonds for ligand A.
      If defined, must be between 0 and 1.
    lambda_angles_A : float | None
      Control parameter for alchemically modified angles for ligand A.
      If defined, must be between 0 and 1.
    lambda_torsions_A : float | None
      Control parameter for alchemically modified dihedrals for ligand A.
      If defined, must be between 0 and 1.
    lambda_restraints_A : float | None
      Control parameter for alchmemically modified restraints for ligand A.

    See Also
    --------
    :class:`openmmtools.states.GlobalParameterState`
    :class:`openfe.protocols.restraint_utils.geometry.DualRegionAlchemicalState`
    """
    lambda_sterics_A = _LambdaParameter("lambda_sterics_A")
    lambda_electrostatics_A = _LambdaParameter("lambda_electrostatics_A")
    lambda_bonds_A = _LambdaParameter("lambda_bonds_A")
    lambda_angles_A = _LambdaParameter("lambda_angles_A")
    lambda_torsions_A = _LambdaParameter("lambda_torsions_A")
    lambda_restraints_A = _LambdaParameter("lambda_restraints_A")


class DualRegionAlchemicalState(SingleRegionAlchemicalState):
    """
    Composable state to control lambda parameters for a system
    with two alchemical molecules / regions (``ligand_A`` and ``ligand_B``).

    Parameters
    ----------
    parameters_name_suffix : str | None
      If specified, the state will control a modified version of the parameter
      ``lambda_restraints_A_{parameters_name_suffix}` instead of just
      ``lambda_restraints_A``.
    lambda_sterics_A : float | None
      Control parameter for the vdW interactions for ligand A.
      If defined, must be between 0 and 1.
    lambda_electrostatics_A : float | None
      Control parameter for the electrostatics interactions for ligand A.
      If defined, must be between 0 and 1.
    lambda_bonds_A : float | None
      Control parameter for alchemically modified bonds for ligand A.
      If defined, must be between 0 and 1.
    lambda_angles_A : float | None
      Control parameter for alchemically modified angles for ligand A.
      If defined, must be between 0 and 1.
    lambda_torsions_A : float | None
      Control parameter for alchemically modified dihedrals for ligand A.
      If defined, must be between 0 and 1.
    lambda_restraints_A : float | None
      Control parameter for alchmemically modified restraints for ligand A.
    lambda_sterics_B : float | None
      Control parameter for the vdW interactions for ligand B.
      If defined, must be between 0 and 1.
    lambda_electrostatics_B : float | None
      Control parameter for the electrostatics interactions for ligand B.
      If defined, must be between 0 and 1.
    lambda_bonds_B : float | None
      Control parameter for alchemically modified bonds for ligand B.
      If defined, must be between 0 and 1.
    lambda_angles_B : float | None
      Control parameter for alchemically modified angles for ligand B.
      If defined, must be between 0 and 1.
    lambda_torsions_B : float | None
      Control parameter for alchemically modified dihedrals for ligand B.
      If defined, must be between 0 and 1.
    lambda_restraints_B : float | None
      Control parameter for alchmemically modified restraints for ligand B.

    See Also
    --------
    :class:`openmmtools.states.GlobalParameterState`
    :class:`openfe.protocols.restraint_utils.geometry.SingleRegionAlchemicalState`
    """
    lambda_sterics_B = _LambdaParameter("lambda_sterics_B")
    lambda_electrostatics_B = _LambdaParameter("lambda_electrostatics_B")
    lambda_bonds_B = _LambdaParameter("lambda_bonds_B")
    lambda_angles_B = _LambdaParameter("lambda_angles_B")
    lambda_torsions_B = _LambdaParameter("lambda_torsions_B")
    lambda_restraints_B = _LambdaParameter("lambda_restraints_B")
