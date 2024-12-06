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
from typing import Optional, Union, Callable

import numpy as np
import openmm
from openmm import unit as omm_unit
from openmmtools.forces import (
    HarmonicRestraintForce,
    HarmonicRestraintBondForce,
    FlatBottomRestraintForce,
    FlatBottomRestraintBondForce,
)
from openmmtools.states import GlobalParameterState, ThermodynamicState
from openff.units.openmm import to_openmm

from gufe.settings.models import SettingsBaseModel
from openfe.protocols.openmm_utils.omm_forces import (
    get_custom_compound_bond_force,
    add_force_in_separate_group,
    get_boresch_energy_function,
    get_periodic_boresch_energy_function,
)


class BaseRestraintGeometry:
    pass


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

    lambda_restraints = GlobalParameterState.GlobalParameter(
        "lambda_restraints", standard_value=1.0
    )

    @lambda_restraints.validator
    def lambda_restraints(self, instance, new_value):
        if new_value is not None and not (0.0 <= new_value <= 1.0):
            errmsg = (
                "lambda_restraints must be between 0.0 and 1.0, " f"got {new_value}"
            )
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

    def __init__(
        self,
        host_atoms: list[int],
        guest_atoms: list[int],
        restraint_settings: SettingsBaseModel,
        restraint_geometry: BaseRestraintGeometry,
        controlling_parameter_name: str = "lambda_restraints",
    ):
        self.host_atoms = host_atoms
        self.guest_atoms = guest_atoms
        self.settings = restraint_settings
        self.geometry = restraint_geometry
        self._verify_input()

    @abc.abstractmethod
    def _verify_inputs(self):
        pass

    @abc.abstractmethod
    def add_force(self, thermodynamic_state: ThermodynamicState):
        pass

    @abc.abstractmethod
    def get_standard_state_correction(self, thermodynamic_state: ThermodynamicState):
        pass

    @abc.abstractmethod
    def _get_force(self):
        pass


class SingleBondMixin:
    def _verify_input(self):
        if len(self.host_atoms) != 1 or len(self.guest_atoms) != 1:
            errmsg = (
                "host_atoms and guest_atoms must only include a single index "
                f"each, got {len(host_atoms)} and "
                f"{len(guest_atoms)} respectively."
            )
            raise ValueError(errmsg)
        super()._verify_inputs()


class BaseRadialllySymmetricRestraintForce(BaseHostGuestRestraints):
    def _verify_inputs(self) -> None:
        if not isinstance(self.settings, BaseDistanceRestraintSettings):
            errmsg = f"Incorrect settings type {self.settings} passed through"
            raise ValueError(errmsg)
        if not isinstance(self.geometry, DistanceRestraintGeometry):
            errmsg = f"Incorrect geometry type {self.geometry} passed through"
            raise ValueError(errmsg)

    def add_force(self, thermodynamic_state: ThermodynamicState) -> None:
        force = self._get_force()
        force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)
        # Note .system is a call to get_system() so it's returning a copy
        system = thermodynamic_state.system
        add_force_in_separate_group(system, force)
        thermodynamic_state.system = system

     def get_standard_state_correction(
        self, thermodynamic_state: ThermodynamicState
    ) -> float:
        force = self._get_force()
        return force.compute_standard_state_correction(
            thermodynamic_state, volume="system"
        )

    def _get_force(self):
        raise NotImplementedError("only implemented in child classes")


class HarmonicBondRestraint(BaseRadialllySymmetricRestraintForce, SingleBondMixin):
    def _get_force(self) -> openmm.Force:
        spring_constant = to_openmm(self.settings.sprint_constant).value_in_unit_system(omm_unit.md_unit_system)
        return HarmonicRestraintBondForce(
            spring_constant=spring_constant,
            restrained_atom_index1=self.host_atoms[0],
            restrained_atom_index2=self.guest_atoms[0],
            controlling_parameter_name=self.controlling_parameter_name,
        )


class FlatBottomBondRestraint(BaseRadialllySymmetricRestraintForce, SingleBondMixin):
    def _get_force(self) -> openmm.Force:
        spring_constant = to_openmm(self.settings.sprint_constant).value_in_unit_system(omm_unit.md_unit_system)
        well_radius = to_openmm(self.settings.well_radius).value_in_unit_system(omm_unit.md_unit_system)
        return FlatBottomRestraintBondForce(
            spring_constant=spring_constant,
            well_radius=well_radius,
            restrained_atom_index1=self.host_atoms[0],
            restrained_atom_index2=self.guest_atoms[0],
            controlling_parameter_name=self.controlling_parameter_name,
        )


class CentroidHarmonicRestraint(BaseRadialllySymmetricRestraintForce):
    def _get_force(self) -> openmm.Force:
        spring_constant = to_openmm(self.settings.sprint_constant).value_in_unit_system(omm_unit.md_unit_system)
        return HarmonicRestraintForce(
            spring_constant=spring_constant,
            restrained_atom_index1=self.host_atoms,
            restrained_atom_index2=self.guest_atoms,
            controlling_parameter_name=self.controlling_parameter_name,
        )


class CentroidFlatBottomRestraint(BaseRadialllySymmetricRestraintForce):
    def _get_force(self):
        spring_constant = to_openmm(self.settings.sprint_constant).value_in_unit_system(omm_unit.md_unit_system)
        well_radius = to_openmm(self.settings.well_radius).value_in_unit_system(omm_unit.md_unit_system)
        return FlatBottomRestraintBondForce(
            spring_constant=spring_constant,
            well_radius=well_radius,
            restrained_atom_index1=self.host_atoms,
            restrained_atom_index2=self.guest_atoms,
            controlling_parameter_name=self.controlling_parameter_name,
        )


class BoreschRestraint(BaseHostGuestRestraints):
    _EFUNC_METHOD: Callable = get_boresch_energy_function
    def _verify_inputs(self) -> None:
        if not isinstance(self.settings, BoreschRestraintSettings):
            errmsg = f"Incorrect settings type {self.settings} passed through"
            raise ValueError(errmsg)
        if not isinstance(self.geometry, BoreschRestraintGeometry):
            errmsg = f"Incorrect geometry type {self.geometry} passed through"
            raise ValueError(errmsg)

    def add_force(self, thermodynamic_state: ThermodynamicState) -> None:
        force = self._get_force()
        force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)
        # Note .system is a call to get_system() so it's returning a copy
        system = thermodynamic_state.system
        add_force_in_separate_group(system, force)
        thermodynamic_state.system = system

    def _get_force(self) -> openmm.Force:
        efunc = _EFUNC_METHOD(
            self.controlling_parameter_name,
        )

        force = get_custom_compound_bond_force(
            n_particles=6, energy_function=efunc
        )

        param_values = []

        parameter_dict = {
            'K_r': self.settings.K_r,
            'r_aA0': self.geometry.r_aA0,
            'K_thetaA': self.settings.K_thetaA,
            'theta_A0': self.geometry.theta_A0,
            'K_thetaB': self.settings.K_thetaB,
            'theta_B0': self.geometry.theta_B0,
            'K_phiA': self.settings.K_phiA,
            'phi_A0': self.geometry.phi_A0,
            'K_phiB': self.settings.K_phiB,
            'phi_B0': self.geometry.phi_B0,
            'K_phiC': self.settings.K_phiC,
            'phi_C0': self.geometry.phi_C0,
        }
        for key, val in parameter_dict.items():
            param_values.append(to_openmm(val).value_in_unit_system(omm_unit.md_unit_system))
            force.addPerBondParameter(key)

        force.addGlobalParameter(self.controlling_parameter_name, 1.0)
        force.addBond(self.host_atoms + self.guest_atoms, param_values)
        return force

    def get_standard_state_correction(
        self, thermodynamic_state: ThermodynamicState
    ) -> float:

        StandardV = 1660.53928 * unit.angstroms**3
        kt = from_openmm(thermodynamic_state.kT)

        # distances
        r_aA0 = self.geometry.r_aA0.to('nm')
        sin_thetaA0 = np.sin(self.geometry.theta_A0.to('radians'))
        sin_thetaB0 = np.sin(self.geometry.theta_B0.to('radians'))

        # restraint energies
        K_r = self.settings.K_r.to('kilojoule_per_mole')
        K_thetaA = self.settings.K_thetaA.to('kilojoule_per_mole')
        k_thetaB = self.settings.K_thetaB.to('kilojoule_per_mole')
        K_phiA = self.settings.K_phiA.to('kilojoule_per_mole')
        K_phiB = self.settings.K_phiB.to('kilojoule_per_mole')
        K_phiC = self.settings.K_phiC.to('kilojoule_per_mole')

        numerator1 = 8.0 * (np.pi**2) * StandardV
        denum1 = (r_aA0**2) * sin_thetaA0 * sin_thetaB0
        numerator2 = np.sqrt(K_r * K_thetaA * K_thetaB * K_phiA * K_phiB * K_phiC)
        denum2 = (2.0 * np.pi * kt)**3

        dG = -kt * np.log((numerator1/denum1) * (numerator2/denum2))

        return dG


# TODO - implement periodic torsion Boresch restraint
