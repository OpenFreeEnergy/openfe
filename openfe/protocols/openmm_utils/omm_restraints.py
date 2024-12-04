# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Classes for applying restraints to OpenMM Systems.

Acknowledgements
----------------
Many of the classes here are at least in part inspired, if not taken from
`Yank <https://github.com/choderalab/yank/>`_ and
`OpenMMTools <https://github.com/choderalab/openmmtools>`_.

TODO
----
* Add relevant duecredit entries.
"""
import abc
from typing import Optional, Union

from openmmtools.states import GlobalParameterState


class RestraintParameterState(GlobalParameterState):
    """
    Composable state to control `lambda_restraints` OpenMM Force parameters.

    See :class:`openmmtools.states.GlobalParameterState` for more details.

    Parameters
    ----------
    parameters_name_suffix : Optional[str]
      If specified, the state will control a modified version of the parameter
      ``lambda_restraints_{parameters_name_suffix}` instead of just ``lambda_restraints``.
    lambda_restraints : Optional[float]
      The strength of the restraint. If defined, must be between 0 and 1.

    Acknowledgement
    ---------------
    Partially reproduced from Yank.
    """

    lambda_restraints = GlobalParameterState.GlobalParameter('lambda_restraints', standard_value=1.0)

    @lambda_restraints.validator
    def lambda_restraints(self, instance, new_value):
        if new_value is not None and not (0.0 <= new_value <= 1.0):
            errmsg = ("lambda_restraints must be between 0.0 and 1.0, "
                      f"got {new_value}")
            raise ValueError(errmsg)
        # Not crashing out on None to match upstream behaviour
        return new_value


class BaseHostGuestRestraints(abc.ABC):
    """
    An abstract base class for defining objects that apply a restraint between
    two entities (referred to as a Host and a Guest).


    TODO
    ----
    Add some examples here.
    """
    def __init__(self, host_atoms: list[int], guest_atoms: list[int], restraint_settings: SettingBaseModel, restraint_geometry: BaseRestraintGeometry):

