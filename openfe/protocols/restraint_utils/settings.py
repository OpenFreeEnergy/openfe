# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Settings for adding restraints.

TODO
----
* Rename from host/guest to molA/molB?
"""
from typing import Optional, Literal
from openff.units import unit
from openff.models.types import FloatQuantity, ArrayQuantity
from pydantic.v1 import validator
from gufe.settings import (
    SettingsBaseModel,
)


class BaseRestraintSettings(SettingsBaseModel):
    """
    Base class for RestraintSettings objects.
    """
    class Config:
        arbitrary_types_allowed = True


class DistanceRestraintSettings(BaseRestraintSettings):
    """
    Settings defining a distance restraint between
    two groups of atoms defined as ``host`` and ``guest``.
    """
    spring_constant: FloatQuantity['kilojoule_per_mole / nm ** 2']
    """
    The distance restraint potential spring constant.
    """
    host_atoms: Optional[list[int]] = None
    """
    The indices of the host component atoms to restrain.
    If defined, these will override any automatic selection.
    """
    guest_atoms: Optional[list[int]] = None
    """
    The indices of the guest component atoms to restraint.
    If defined, these will override any automatic selection.
    """
    central_atoms_only: bool = False
    """
    Whether to apply the restraint solely to the central atoms
    of each group.

    Note: this can only be applied if ``host`` and ``guest``
    represent small molecules.
    """

    @validator("guest_atoms", "host_atoms")
    def positive_idxs(cls, v):
        if v is not None and any([i < 0 for i in v]):
            errmsg = "negative indices passed"
            raise ValueError(errmsg)
        return v


class FlatBottomRestraintSettings(DistanceRestraintSettings):
    """
    Settings to define a flat bottom restraint between two
    groups of atoms named ``host`` and ``guest``.
    """
    well_radius: Optional[FloatQuantity['nm']] = None
    """
    The distance at which the harmonic restraint is imposed
    in units of distance.
    """
    @validator("well_radius")
    def positive_value(cls, v):
        if v is not None and v.m < 0:
            errmsg = f"well radius cannot be negative {v}"
            raise ValueError(errmsg)
        return v


class BoreschRestraintSettings(BaseRestraintSettings):
    """
    Settings to define a Boresch-style restraint between
    two groups of atoms named ``host`` and ``guest``.

    The restraint is defined in the following manner:

      H2                         G2
       -                        -
        -                      -
         H1 - - H0 -- G0 - - G1

    Where HX represents the X index of ``host_atoms``
    and GX the X indexx of ``guest_atoms``.

    By default, the Boresch-like restraint will be
    obtained using a modified version of the
    search algorithm implemented by Baumann et al. [1].

    If ``guest_atoms`` and ``host_atoms`` are defined,
    these indices will be used instead.

    References
    ----------
    [1] Baumann, Hannah M., et al. "Broadening the scope of binding free
        energy calculations using a Separated Topologies approach." (2023).
    """
    K_r: FloatQuantity['kilojoule_per_mole / nm ** 2']
    """
    The bond spring constant between H0 and G0.
    """
    K_thetaA: FloatQuantity['kilojoule_per_mole / radians ** 2']
    """
    The spring constant for the angle formed by H1-H0-G0.
    """
    K_thetaB: FloatQuantity['kilojoule_per_mole / radians ** 2']
    """
    The spring constant for the angle formed by H0-G0-G1.
    """
    phi_A0: FloatQuantity['kilojoule_per_mole / radians ** 2']
    """
    The equilibrium force constant for the dihedral formed by
    H2-H1-H0-G0.
    """
    phi_B0: FloatQuantity['kilojoule_per_mole / radians ** 2']
    """
    The equilibrium force constant for the dihedral formed by
    H1-H0-G0-G1.
    """
    phi_C0: FloatQuantity['kilojoule_per_mole / radians ** 2']
    """
    The equilibrium force constant for the dihedral formed by
    H0-G0-G1-G2.
    """
    host_atoms: Optional[list[int]] = None
    """
    The indices of the host component atoms to restrain.
    If defined, these will override any automatic selection.
    """
    guest_atoms: Optional[list[int]] = None
    """
    The indices of the guest component atoms to restraint.
    If defined, these will override any automatic selection.
    """

    @validator("guest_atoms", "host_atoms")
    def positive_idxs_list(cls, v):
        if v is not None and any([i < 0 for i in v]):
            errmsg = "negative indices passed"
            raise ValueError(errmsg)
        return v