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

from openfe.protocols.restraint_utils.geometry import (
    BaseRestraintGeometry,
    DistanceRestraintGeometry,
    BoreschRestraintGeometry
)

from openfe.protocols.restraint_utils.settings import (
    DistanceRestraintSettings,
    BoreschRestraintSettings,
)

from .omm_forces import (
    get_custom_compound_bond_force,
    add_force_in_separate_group,
    get_boresch_energy_function,
)


class RestraintParameterState(GlobalParameterState):
    """
    Composable state to control `lambda_restraints` OpenMM Force parameters.

    See :class:`openmmtools.states.GlobalParameterState` for more details.

    Parameters
    ----------
    parameters_name_suffix : Optional[str]
      If specified, the state will control a modified version of the parameter
      ``lambda_restraints_{parameters_name_suffix}` instead of just
      ``lambda_restraints``.
    lambda_restraints : Optional[float]
      The scaling parameter for the restraint. If defined,
      must be between 0 and 1. In most cases, a value of 1 indicates that the
      restraint is fully turned on, whilst a value of 0 indicates that it is
      innactive.

    Acknowledgement
    ---------------
    Partially reproduced from Yank.
    """
    # We set the standard system to a fully interacting restraint
    lambda_restraints = GlobalParameterState.GlobalParameter(
        "lambda_restraints", standard_value=1.0
    )

    @lambda_restraints.validator
    def lambda_restraints(self, instance, new_value):
        if new_value is not None and not (0.0 <= new_value <= 1.0):
            errmsg = (
                "lambda_restraints must be between 0.0 and 1.0 "
                f"and got {new_value}"
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
    Add some developer examples here.
    """

    def __init__(
        self,
        restraint_settings: SettingsBaseModel,
    ):
        self.settings = restraint_settings
        self._verify_inputs()

    @abc.abstractmethod
    def _verify_inputs(self):
        """
        Method for validating that the inputs to the class are correct.
        """
        pass

    @abc.abstractmethod
    def _verify_geometry(self, geometry):
        """
        Method for validating that the geometry object passed is correct.
        """
        pass

    @abc.abstractmethod
    def add_force(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: BaseRestraintGeometry,
        controlling_parameter_name: str,
    ):
        """
        Method for in-place adding the Force to the System of a
        ThermodynamicState.

        Parameters
        ----------
        thermodymamic_state : ThermodynamicState
          The ThermodynamicState with a System to inplace modify with the
          new force.
        geometry : BaseRestraintGeometry
          A geometry object defining the restraint parameters.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.
        """
        pass

    @abc.abstractmethod
    def get_standard_state_correction(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: BaseRestraintGeometry
    ) -> unit.Quantity:
        """
        Get the standard state correction for the Force when
        applied to the input ThermodynamicState.

        Parameters
        ----------
        thermodymamic_state : ThermodynamicState
          The ThermodynamicState with a System to inplace modify with the
          new force.
        geometry : BaseRestraintGeometry
          A geometry object defining the restraint parameters.

        Returns
        -------
        correction : unit.Quantity
          The standard state correction free energy in units compatible
          with kilojoule per mole.
        """
        pass

    @abc.abstractmethod
    def _get_force(
        self,
        geometry: BaseRestraintGeometry,
        controlling_parameter_name: str,
    ):
        """
        Helper method to get the relevant OpenMM Force for this
        class, given an input geometry.
        """
        pass


class SingleBondMixin:
    """
    A mixin to extend geometry checks for Forces that can only hold
    a single atom.
    """
    def _verify_geometry(self, geometry: BaseRestraintGeometry):
        if len(geometry.host_atoms) != 1 or len(geometry.guest_atoms) != 1:
            errmsg = (
                "host_atoms and guest_atoms must only include a single index "
                f"each, got {len(geometry.host_atoms)} and "
                f"{len(geometry.guest_atoms)} respectively."
            )
            raise ValueError(errmsg)
        super()._verify_geometry(geometry)


class BaseRadiallySymmetricRestraintForce(BaseHostGuestRestraints):
    """
    A base class for all radially symmetic Forces acting between
    two sets of atoms.

    Must be subclassed.
    """
    def _verify_inputs(self) -> None:
        if not isinstance(self.settings, DistanceRestraintSettings):
            errmsg = f"Incorrect settings type {self.settings} passed through"
            raise ValueError(errmsg)

    def _verify_geometry(self, geometry: DistanceRestraintGeometry):
        if not isinstance(geometry, DistanceRestraintGeometry):
            errmsg = f"Incorrect geometry class type {geometry} passed through"
            raise ValueError(errmsg)

    def add_force(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: DistanceRestraintGeometry,
        controlling_parameter_name: str = "lambda_restraints",
    ) -> None:
        """
        Method for in-place adding the Force to the System of the
        given ThermodynamicState.

        Parameters
        ----------
        thermodymamic_state : ThermodynamicState
          The ThermodynamicState with a System to inplace modify with the
          new force.
        geometry : BaseRestraintGeometry
          A geometry object defining the restraint parameters.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.
        """
        self._verify_geometry(geometry)
        force = self._get_force(geometry, controlling_parameter_name)
        force.setUsesPeriodicBoundaryConditions(
            thermodynamic_state.is_periodic
        )
        # Note .system is a call to get_system() so it's returning a copy
        system = thermodynamic_state.system
        add_force_in_separate_group(system, force)
        thermodynamic_state.system = system

    def get_standard_state_correction(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: DistanceRestraintGeometry,
    ) -> unit.Quantity:
        """
        Get the standard state correction for the Force when
        applied to the input ThermodynamicState.

        Parameters
        ----------
        thermodymamic_state : ThermodynamicState
          The ThermodynamicState with a System to inplace modify with the
          new force.
        geometry : BaseRestraintGeometry
          A geometry object defining the restraint parameters.

        Returns
        -------
        correction : unit.Quantity
          The standard state correction free energy in units compatible
          with kilojoule per mole.
        """
        self._verify_geometry(geometry)
        force = self._get_force(geometry)
        corr = force.compute_standard_state_correction(
            thermodynamic_state, volume="system"
        )
        dg = corr * thermodynamic_state.kT
        return from_openmm(dg).to('kilojoule_per_mole')

    def _get_force(
        self,
        geometry: DistanceRestraintGeometry,
        controlling_parameter_name: str
    ):
        raise NotImplementedError("only implemented in child classes")


class HarmonicBondRestraint(
    BaseRadiallySymmetricRestraintForce, SingleBondMixin
):
    """
    A class to add a harmonic restraint between two atoms
    in an OpenMM system.

    The restraint is defined as a
    :class:`openmmtools.forces.HarmonicRestraintBondForce`.

    Notes
    -----
    * Settings must contain a ``spring_constant`` for the
      Force in units compatible with kilojoule/mole/nm**2.
    """
    def _get_force(
        self,
        geometry: DistanceRestraintGeometry,
        controlling_parameter_name: str,
    ) -> openmm.Force:
        """
        Get the HarmonicRestraintBondForce given an input geometry.

        Parameters
        ----------
        geometry : DistanceRestraintGeometry
          A geometry class that defines how the Force is applied.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.

        Returns
        -------
        HarmonicRestraintBondForce
          An OpenMM Force that applies a harmonic restraint between
          two atoms.
        """
        spring_constant = to_openmm(
            self.settings.spring_constant
        ).value_in_unit_system(omm_unit.md_unit_system)
        return HarmonicRestraintBondForce(
            spring_constant=spring_constant,
            restrained_atom_index1=geometry.host_atoms[0],
            restrained_atom_index2=geometry.guest_atoms[0],
            controlling_parameter_name=controlling_parameter_name,
        )


class FlatBottomBondRestraint(
    BaseRadiallySymmetricRestraintForce, SingleBondMixin
):
    """
    A class to add a flat bottom restraint between two atoms
    in an OpenMM system.

    The restraint is defined as a
    :class:`openmmtools.forces.FlatBottomRestraintBondForce`.

    Notes
    -----
    * Settings must contain a ``spring_constant`` for the
      Force in units compatible with kilojoule/mole/nm**2.
    """
    def _get_force(
        self,
        geometry: DistanceRestraintGeometry,
        controlling_parameter_name: str,
    ) -> openmm.Force:
        """
        Get the FlatBottomRestraintBondForce given an input geometry.

        Parameters
        ----------
        geometry : DistanceRestraintGeometry
          A geometry class that defines how the Force is applied.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.

        Returns
        -------
        FlatBottomRestraintBondForce
          An OpenMM Force that applies a flat bottom restraint between
          two atoms.
        """
        spring_constant = to_openmm(
            self.settings.spring_constant
        ).value_in_unit_system(omm_unit.md_unit_system)
        well_radius = to_openmm(
            geometry.well_radius
        ).value_in_unit_system(omm_unit.md_unit_system)
        return FlatBottomRestraintBondForce(
            spring_constant=spring_constant,
            well_radius=well_radius,
            restrained_atom_index1=geometry.host_atoms[0],
            restrained_atom_index2=geometry.guest_atoms[0],
            controlling_parameter_name=controlling_parameter_name,
        )


class CentroidHarmonicRestraint(BaseRadiallySymmetricRestraintForce):
    """
    A class to add a harmonic restraint between the centroid of
    two sets of atoms in an OpenMM system.

    The restraint is defined as a
    :class:`openmmtools.forces.HarmonicRestraintForce`.

    Notes
    -----
    * Settings must contain a ``spring_constant`` for the
      Force in units compatible with kilojoule/mole/nm**2.
    """
    def _get_force(
        self,
        geometry: DistanceRestraintGeometry,
        controlling_parameter_name: str,
    ) -> openmm.Force:
        """
        Get the HarmonicRestraintForce given an input geometry.

        Parameters
        ----------
        geometry : DistanceRestraintGeometry
          A geometry class that defines how the Force is applied.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.

        Returns
        -------
        HarmonicRestraintForce
          An OpenMM Force that applies a harmonic restraint between
          the centroid of two sets of atoms.
        """
        spring_constant = to_openmm(
            self.settings.spring_constant
        ).value_in_unit_system(omm_unit.md_unit_system)
        return HarmonicRestraintForce(
            spring_constant=spring_constant,
            restrained_atom_index1=geometry.host_atoms,
            restrained_atom_index2=geometry.guest_atoms,
            controlling_parameter_name=controlling_parameter_name,
        )


class CentroidFlatBottomRestraint(BaseRadiallySymmetricRestraintForce):
    """
    A class to add a flat bottom restraint between the centroid
    of two sets of atoms in an OpenMM system.

    The restraint is defined as a
    :class:`openmmtools.forces.FlatBottomRestraintForce`.

    Notes
    -----
    * Settings must contain a ``spring_constant`` for the
      Force in units compatible with kilojoule/mole/nm**2.
    """
    def _get_force(
        self,
        geometry: DistanceRestraintGeometry,
        controlling_parameter_name: str,
    ) -> openmm.Force:
        """
        Get the FlatBottomRestraintForce given an input geometry.

        Parameters
        ----------
        geometry : DistanceRestraintGeometry
          A geometry class that defines how the Force is applied.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.

        Returns
        -------
        FlatBottomRestraintForce
          An OpenMM Force that applies a flat bottom restraint between
          the centroid of two sets of atoms.
        """
        spring_constant = to_openmm(
            self.settings.spring_constant
        ).value_in_unit_system(omm_unit.md_unit_system)
        well_radius = to_openmm(
            geometry.well_radius
        ).value_in_unit_system(omm_unit.md_unit_system)
        return FlatBottomRestraintForce(
            spring_constant=spring_constant,
            well_radius=well_radius,
            restrained_atom_index1=geometry.host_atoms,
            restrained_atom_index2=geometry.guest_atoms,
            controlling_parameter_name=controlling_parameter_name,
        )


class BoreschRestraint(BaseHostGuestRestraints):
    """
    A class to add a Boresch-like restraint between six atoms,

    The restraint is defined as a
    :class:`openmmtools.forces.CustomCompoundForce` with the
    following energy function:

        lambda_control_parameter * E;
        E = (K_r/2)*(distance(p3,p4) - r_aA0)^2
        + (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2
        + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2
        + (K_phiA/2)*dphi_A^2 + (K_phiB/2)*dphi_B^2
        + (K_phiC/2)*dphi_C^2;
        dphi_A = dA - floor(dA/(2.0*pi)+0.5)*(2.0*pi);
        dA = dihedral(p1,p2,p3,p4) - phi_A0;
        dphi_B = dB - floor(dB/(2.0*pi)+0.5)*(2.0*pi);
        dB = dihedral(p2,p3,p4,p5) - phi_B0;
        dphi_C = dC - floor(dC/(2.0*pi)+0.5)*(2.0*pi);
        dC = dihedral(p3,p4,p5,p6) - phi_C0;

    Where p1, p2, p3, p4, p5, p6 represent host atoms 2, 1, 0,
    and guest atoms 0, 1, 2 respectively.

    ``lambda_control_parameter`` is a control parameter for
    scaling the Force.

    ``K_r`` is defined as the bond spring constant between
    p3 and p4 and must be provided in the settings in units
    compatible with kilojoule / mole.

    ``r_aA0`` is the equilibrium distance of the bond between
    p3 and p4. This must be provided by the Geometry class in
    units compatiblle with nanometer.

    ``K_thetaA`` and ``K_thetaB`` are the spring constants for the angles
    formed by (p2, p3, p4) and (p3, p4, p5). They must be provided in the
    settings in units compatible with kilojoule / mole / radians**2.

    ``theta_A0`` and ``theta_B0`` are the equilibrium values for angles
    (p2, p3, p4) and (p3, p4, p5). They must be provided by the
    Geometry class in units compatible with radians.

    ``phi_A0``, ``phi_B0``, and ``phi_C0`` are the equilibrium force constants
    for the dihedrals formed by (p1, p2, p3, p4), (p2, p3, p4, p5), and
    (p3, p4, p5, p6). They must be provided in the settings in units
    compatible with kilojoule / mole / radians ** 2.

    ``phi_A0``, ``phi_B0``, and ``phi_C0`` are the equilibrium values
    for the dihedrals formed by (p1, p2, p3, p4), (p2, p3, p4, p5), and
    (p3, p4, p5, p6). They must be provided in the Geometry class in
    units compatible with radians.


    Notes
    -----
    * Settings must define the ``K_r`` (d)
    """
    def _verify_inputs(self) -> None:
        """
        Method for validating that the geometry object is correct.
        """
        if not isinstance(self.settings, BoreschRestraintSettings):
            errmsg = f"Incorrect settings type {self.settings} passed through"
            raise ValueError(errmsg)

    def _verify_geometry(self, geometry: BoreschRestraintGeometry):
        """
        Method for validating that the geometry object is correct.
        """
        if not isinstance(geometry, BoreschRestraintGeometry):
            errmsg = f"Incorrect geometry class type {geometry} passed through"
            raise ValueError(errmsg)

    def add_force(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: BoreschRestraintGeometry,
        controlling_parameter_name: str,
    ) -> None:
        """
        Method for in-place adding the Boresch CustomCompoundForce
        to the System of the given ThermodynamicState.

        Parameters
        ----------
        thermodymamic_state : ThermodynamicState
          The ThermodynamicState with a System to inplace modify with the
          new force.
        geometry : BaseRestraintGeometry
          A geometry object defining the restraint parameters.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.
        """
        self._verify_geometry(geometry)
        force = self._get_force(
            geometry,
            controlling_parameter_name,
        )
        force.setUsesPeriodicBoundaryConditions(
            thermodynamic_state.is_periodic
        )
        # Note .system is a call to get_system() so it's returning a copy
        system = thermodynamic_state.system
        add_force_in_separate_group(system, force)
        thermodynamic_state.system = system

    def _get_force(
        self,
        geometry: BoreschRestraintGeometry,
        controlling_parameter_name: str
    ) -> openmm.CustomCompoundBondForce:
        """
        Get the CustomCompoundForce with a Boresch-like energy function
        given an input geometry.

        Parameters
        ----------
        geometry : DistanceRestraintGeometry
          A geometry class that defines how the Force is applied.
        controlling_parameter_name : str
          The name of the controlling parameter for the Force.

        Returns
        -------
        CustomCompoundForce
          An OpenMM CustomCompoundForce that applies a Boresch-like
          restraint between 6 atoms.
        """
        efunc = get_boresch_energy_function(controlling_parameter_name)

        force = get_custom_compound_bond_force(
            energy_function=efunc, n_particles=6,
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
            param_values.append(
                to_openmm(val).value_in_unit_system(omm_unit.md_unit_system)
            )
            force.addPerBondParameter(key)

        force.addGlobalParameter(controlling_parameter_name, 1.0)
        atoms = [
            geometry.host_atoms[2],
            geometry.host_atoms[1],
            geometry.host_atoms[0],
            geometry.guest_atoms[0],
            geometry.guest_atoms[1],
            geometry.guest_atoms[2],
        ]
        force.addBond(atoms, param_values)
        return force

    def get_standard_state_correction(
        self,
        thermodynamic_state: ThermodynamicState,
        geometry: BoreschRestraintGeometry
    ) -> unit.Quantity:
        """
        Get the standard state correction for the Boresch-like
        restraint when applied to the input ThermodynamicState.

        The correction is calculated using the analytical method
        as defined by Boresch et al. [1]

        Parameters
        ----------
        thermodymamic_state : ThermodynamicState
          The ThermodynamicState with a System to inplace modify with the
          new force.
        geometry : BaseRestraintGeometry
          A geometry object defining the restraint parameters.

        Returns
        -------
        correction : unit.Quantity
          The standard state correction free energy in units compatible
          with kilojoule per mole.

        References
        ----------
        [1] Boresch S, Tettinger F, Leitgeb M, Karplus M. J Phys Chem B. 107:9535, 2003.
            http://dx.doi.org/10.1021/jp0217839
        """
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
        K_thetaB = self.settings.K_thetaB.to('kilojoule_per_mole / radians ** 2')
        K_phiA = self.settings.K_phiA.to('kilojoule_per_mole / radians ** 2')
        K_phiB = self.settings.K_phiB.to('kilojoule_per_mole / radians ** 2')
        K_phiC = self.settings.K_phiC.to('kilojoule_per_mole / radians ** 2')

        numerator1 = 8.0 * (np.pi**2) * StandardV
        denum1 = (r_aA0**2) * sin_thetaA0 * sin_thetaB0
        numerator2 = np.sqrt(
            K_r * K_thetaA * K_thetaB * K_phiA * K_phiB * K_phiC
        )
        denum2 = (2.0 * np.pi * kt)**3

        dG = -kt * np.log((numerator1/denum1) * (numerator2/denum2))

        return dG
