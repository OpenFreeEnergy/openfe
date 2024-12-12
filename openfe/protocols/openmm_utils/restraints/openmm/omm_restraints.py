# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Classes for applying restraints to OpenMM Systems.

Acknowledgements
----------------
Many of the classes here are at least in part inspired from
`Yank <https://github.com/choderalab/yank/>`_ and
`OpenMMTools <https://github.com/choderalab/openmmtools>`_.

TODO
----
* Add relevant duecredit entries.
* Add Periodic Torsion Boresch class
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
from openff.units.openmm import to_openmm, from_openmm
from openff.units import unit

from gufe.settings.models import SettingsBaseModel
from openfe.protocols.openmm_utils.omm_forces import (
    get_custom_compound_bond_force,
    add_force_in_separate_group,
    get_boresch_energy_function,
    get_periodic_boresch_energy_function,
)


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
        restraint_settings: SettingsBaseModel,
        controlling_parameter_name: str = "lambda_restraints",
    ):
        self.settings = restraint_settings
        self._verify_settings()

    @abc.abstractmethod
    def _verify_settings(self):
        pass

    @abc.abstractmethod
    def _verify_geometry(self, geometry):
        pass

    @abc.abstractmethod
    def add_force(self, thermodynamic_state: ThermodynamicState, geometry: BaseRestraintGeometry):
        pass

    @abc.abstractmethod
    def get_standard_state_correction(self, thermodynamic_state: ThermodynamicState, geometry: BaseRestraintGeometry):
        pass

    @abc.abstractmethod
    def _get_force(self, geometry: BaseRestraintGeometry):
        pass


class SingleBondMixin:
    def _verify_geometry(self, geometry: BaseRestraintGeometry):
        if len(geometry.host_atoms) != 1 or len(geometry.guest_atoms) != 1:
            errmsg = (
                "host_atoms and guest_atoms must only include a single index "
                f"each, got {len(host_atoms)} and "
                f"{len(guest_atoms)} respectively."
            )
            raise ValueError(errmsg)
        super()._verify_geometry(geometry)


class BaseRadiallySymmetricRestraintForce(BaseHostGuestRestraints):
    def _verify_inputs(self) -> None:
        if not isinstance(self.settings, BaseDistanceRestraintSettings):
            errmsg = f"Incorrect settings type {self.settings} passed through"
            raise ValueError(errmsg)

    def _verify_geometry(self, geometry: DistanceRestraintGeometry)
        if not isinstance(geometry, DistanceRestraintGeometry):
            errmsg = f"Incorrect geometry class type {geometry} passed through"
            raise ValueError(errmsg)

    def add_force(self, thermodynamic_state: ThermodynamicState, geometry: DistanceRestraintGeometry) -> None:
        self._verify_geometry(geometry)
        force = self._get_force(geometry)
        force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)
        # Note .system is a call to get_system() so it's returning a copy
        system = thermodynamic_state.system
        add_force_in_separate_group(system, force)
        thermodynamic_state.system = system

    def get_standard_state_correction(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: DistanceRestraintGeometry,
    ) -> unit.Quantity:
        self._verify_geometry(geometry)
        force = self._get_force(geometry)
        corr = force.compute_standard_state_correction(
            thermodynamic_state, volume="system"
        )
        dg = corr * thermodynamic_state.kT
        return from_openmm(dg).to('kilojoule_per_mole')

    def _get_force(self, geometry: DistanceRestraintGeometry):
        raise NotImplementedError("only implemented in child classes")


class HarmonicBondRestraint(BaseRadiallySymmetricRestraintForce, SingleBondMixin):
    def _get_force(self, geometry: DistanceRestraintGeometry) -> openmm.Force:
        spring_constant = to_openmm(self.settings.spring_constant).value_in_unit_system(omm_unit.md_unit_system)
        return HarmonicRestraintBondForce(
            spring_constant=spring_constant,
            restrained_atom_index1=geometry.host_atoms[0],
            restrained_atom_index2=geometry.guest_atoms[0],
            controlling_parameter_name=self.controlling_parameter_name,
        )


class FlatBottomBondRestraint(BaseRadiallySymmetricRestraintForce, SingleBondMixin):
    def _get_force(self, geometry: DistanceRestraintGeometry) -> openmm.Force:
        spring_constant = to_openmm(self.settings.spring_constant).value_in_unit_system(omm_unit.md_unit_system)
        well_radius = to_openmm(geometry.well_radius).value_in_unit_system(omm_unit.md_unit_system)
        return FlatBottomRestraintBondForce(
            spring_constant=spring_constant,
            well_radius=well_radius,
            restrained_atom_index1=geometry.host_atoms[0],
            restrained_atom_index2=geometry.guest_atoms[0],
            controlling_parameter_name=self.controlling_parameter_name,
        )


class CentroidHarmonicRestraint(BaseRadiallySymmetricRestraintForce):
    def _get_force(self, geometry: DistanceRestraintGeometry) -> openmm.Force:
        spring_constant = to_openmm(self.settings.spring_constant).value_in_unit_system(omm_unit.md_unit_system)
        return HarmonicRestraintForce(
            spring_constant=spring_constant,
            restrained_atom_index1=geometry.host_atoms,
            restrained_atom_index2=geometry.guest_atoms,
            controlling_parameter_name=self.controlling_parameter_name,
        )


class CentroidFlatBottomRestraint(BaseRadiallySymmetricRestraintForce):
    def _get_force(self, geometry: DistanceRestraintGeometry) -> openmm.Force:
        spring_constant = to_openmm(self.settings.spring_constant).value_in_unit_system(omm_unit.md_unit_system)
        well_radius = to_openmm(geometry.well_radius).value_in_unit_system(omm_unit.md_unit_system)
        return FlatBottomRestraintBondForce(
            spring_constant=spring_constant,
            well_radius=well_radius,
            restrained_atom_index1=geometry.host_atoms,
            restrained_atom_index2=geometry.guest_atoms,
            controlling_parameter_name=self.controlling_parameter_name,
        )


class BoreschRestraint(BaseHostGuestRestraints):
    def _verify_settings(self) -> None:
        if not isinstance(self.settings, BoreschRestraintSettings):
            errmsg = f"Incorrect settings type {self.settings} passed through"
            raise ValueError(errmsg)

    def _verify_geometry(self, geometry: BoreschRestraintGeometry):
        if not isinstance(geometry, BoreschRestraintGeometry):
            errmsg = f"Incorrect geometry class type {geometry} passed through"
            raise ValueError(errmsg)

    def add_force(self, thermodynamic_state: ThermodynamicState, geometry: BoreschRestraintGeometry) -> None:
        _verify_geometry(geometry)
        force = self._get_force(geometry)
        force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)
        # Note .system is a call to get_system() so it's returning a copy
        system = thermodynamic_state.system
        add_force_in_separate_group(system, force)
        thermodynamic_state.system = system

    def _get_force(self, geometry: BoreschRestraintGeometry) -> openmm.Force:
        efunc = get_boresch_energy_function(
            self.controlling_parameter_name,
        )

        force = get_custom_compound_bond_force(
            n_particles=6, energy_function=efunc
        )

        param_values = []

        parameter_dict = {
            'K_r': self.settings.K_r,
            'r_aA0': geometry.r_aA0,
            'K_thetaA': self.settings.K_thetaA,
            'theta_A0': geometry.theta_A0,
            'K_thetaB': self.settings.K_thetaB,
            'theta_B0': geometry.theta_B0,
            'K_phiA': self.settings.K_phiA,
            'phi_A0': geometry.phi_A0,
            'K_phiB': self.settings.K_phiB,
            'phi_B0': geometry.phi_B0,
            'K_phiC': self.settings.K_phiC,
            'phi_C0': geometry.phi_C0,
        }
        for key, val in parameter_dict.items():
            param_values.append(to_openmm(val).value_in_unit_system(omm_unit.md_unit_system))
            force.addPerBondParameter(key)

        force.addGlobalParameter(self.controlling_parameter_name, 1.0)
        force.addBond(geometry.host_atoms + geometry.guest_atoms, param_values)
        return force

    def get_standard_state_correction(
        self, thermodynamic_state: ThermodynamicState, geometry: BoreschRestraintGeometry
    ) -> unit.Quantity:
        self._verify_geometry(geometry)

        StandardV = 1.66053928 * unit.nanometer**3
        kt = from_openmm(thermodynamic_state.kT)

        # distances
        r_aA0 = geometry.r_aA0.to('nm')
        sin_thetaA0 = np.sin(geometry.theta_A0.to('radians'))
        sin_thetaB0 = np.sin(geometry.theta_B0.to('radians'))

        # restraint energies
        K_r = self.settings.K_r.to('kilojoule_per_mole / nm ** 2')
        K_thetaA = self.settings.K_thetaA.to('kilojoule_per_mole / radians ** 2')
        k_thetaB = self.settings.K_thetaB.to('kilojoule_per_mole / radians ** 2')
        K_phiA = self.settings.K_phiA.to('kilojoule_per_mole / radians ** 2')
        K_phiB = self.settings.K_phiB.to('kilojoule_per_mole / radians ** 2')
        K_phiC = self.settings.K_phiC.to('kilojoule_per_mole / radians ** 2')

        numerator1 = 8.0 * (np.pi**2) * StandardV
        denum1 = (r_aA0**2) * sin_thetaA0 * sin_thetaB0
        numerator2 = np.sqrt(K_r * K_thetaA * K_thetaB * K_phiA * K_phiB * K_phiC)
        denum2 = (2.0 * np.pi * kt)**3

        dG = -kt * np.log((numerator1/denum1) * (numerator2/denum2))

        return dG
