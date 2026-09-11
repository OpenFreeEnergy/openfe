# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Settings for adding restraints.

TODO
----
* Rename from host/guest to molA/molB?
* Add all the restraint settings entries.
"""

from typing import Annotated, Literal, Optional, TypeAlias

from gufe.settings import SettingsBaseModel
from gufe.settings.typing import GufeQuantity, NanometerQuantity, specify_quantity_units
from openff.units import unit
from pydantic import ConfigDict, field_validator, model_validator

SpringConstantLinearQuantity: TypeAlias = Annotated[
    GufeQuantity, specify_quantity_units("kilojoule_per_mole / nm ** 2")
]
SpringConstantAngularQuantity: TypeAlias = Annotated[
    GufeQuantity, specify_quantity_units("kilojoule_per_mole / radians ** 2")
]
AngleQuantity: TypeAlias = Annotated[GufeQuantity, specify_quantity_units("radians")]


class BaseRestraintSettings(SettingsBaseModel):
    """
    Base class for RestraintSettings objects.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DistanceRestraintSettings(BaseRestraintSettings):
    """
    Settings defining a distance restraint between
    two groups of atoms defined as ``host`` and ``guest``.
    """

    spring_constant: SpringConstantLinearQuantity
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

    @field_validator("guest_atoms", "host_atoms")
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

    well_radius: NanometerQuantity | None = None
    """
    The distance at which the harmonic restraint is imposed
    in units of distance.
    """

    @field_validator("well_radius")
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
    [2] Wu, Zhiyi, et al. "Optimizing Absolute Binding Free Energy
        Calculations for Production Usage."
        (2025; DOI 10.26434/chemrxiv-2025-q08ld-v2)
    """

    K_r: SpringConstantLinearQuantity = 4184.0 * unit.kilojoule_per_mole / unit.nm**2
    """
    The bond spring constant between H0 and G0. Default 10 kcal/mol/Å²
    """
    K_thetaA: SpringConstantAngularQuantity = 334.72 * unit.kilojoule_per_mole / unit.radians**2
    """
    The spring constant for the angle formed by H1-H0-G0.
    Default 80 kcal/mol/rad²
    """
    K_thetaB: SpringConstantAngularQuantity = 334.72 * unit.kilojoule_per_mole / unit.radians**2
    """
    The spring constant for the angle formed by H0-G0-G1.
    Default 80 kcal/mol/rad²
    """
    K_phiA: SpringConstantAngularQuantity = 334.72 * unit.kilojoule_per_mole / unit.radians**2
    """
    The equilibrium force constant for the dihedral formed by
    H2-H1-H0-G0. Default 80 kcal/mol/rad²
    """
    K_phiB: SpringConstantAngularQuantity = 334.72 * unit.kilojoule_per_mole / unit.radians**2
    """
    The equilibrium force constant for the dihedral formed by
    H1-H0-G0-G1. Default 80 kcal/mol/rad²
    """
    K_phiC: SpringConstantAngularQuantity = 334.72 * unit.kilojoule_per_mole / unit.radians**2
    """
    The equilibrium force constant for the dihedral formed by
    H0-G0-G1-G2. Default 80 kcal/mol/rad²
    """
    host_selection: str = "backbone"
    """
    Boresch-like restraint search parameter.
    An MDAnalysis selection string to sub-select the host atoms which will be involved in the restraint.
    """
    dssp_filter: bool = True
    """
    Boresch-like restraint search parameter.
    Whether or not to try to do a DSSP filter on the host atoms.
    """
    rmsf_cutoff: NanometerQuantity = 0.1 * unit.nanometer
    """
    Boresch-like restraint search parameter.
    The cutoff value for filtering atoms by their root mean square fluctuation. Atoms with values above this cutoff will be disregarded.
    """
    host_min_distance: NanometerQuantity = 0.5 * unit.nanometer
    """
    Boresch-like restraint search parameter.
    The minimum distance between any host atom and the guest G0 atom. Must be in units compatible with nanometer.
    """
    host_max_distance: NanometerQuantity = 1.5 * unit.nanometer
    """
    Boresch-like restraint search parameter.
    The maximum distance between any host atom and the guest G0 atom. Must be in units compatible with nanometer.
    """
    # TODO: re-enable this (Issue #1555)
    # host_atoms: Optional[list[int]] = None
    # """
    # The indices of the host component atoms to restrain.
    # If defined, these will override any automatic selection.
    # """
    # guest_atoms: Optional[list[int]] = None
    # """
    # The indices of the guest component atoms to restraint.
    # If defined, these will override any automatic selection.
    # """
    anchor_finding_strategy: Literal["multi-residue", "bonded"] = "bonded"
    """
    The Boresch atom picking strategy to use.

    Current options:
      * `bonded`: pick host atoms that are bonded to each other.
      * `multi-residue`: pick host atoms which can span multiple residues.
    """


#     @field_validator("guest_atoms", "host_atoms")
#     def positive_idxs_list(cls, v):
#         if v is not None and any([i < 0 for i in v]):
#             errmsg = "negative indices passed"
#             raise ValueError(errmsg)
#         return v

class DihedralRestraintSettings(BaseRestraintSettings):
    """
    Settings to define flat-bottomed harmonic restraints on a set of a
    ligand's own dihedrals.

    The restraint is intended to be fully off in the interacting end state and
    fully on in the non-interacting end state, holding the decoupled ligand in
    the conformation of its input pose. Because a decoupled ligand does not
    interact with its environment, the free energy of applying the restraint
    in the non-interacting end state is a property of the isolated molecule
    alone. Applied identically in the complex and solvent legs, the
    contribution therefore cancels in the resulting ddG and no standard state
    correction is required.
    """

    spring_constant: SpringConstantAngularQuantity = (
        334.72 * unit.kilojoule_per_mole / unit.radians**2
    )
    """
    The dihedral restraint potential spring constant, applied beyond
    ``half_width`` of the target angle.
    """
    half_width: AngleQuantity = 0.5235987755982988 * unit.radians
    """
    The half width of the flat-bottomed well, i.e. the dihedral is unrestrained
    within this angle of its target. Set to zero for a purely harmonic
    restraint. Default 30 degrees, which is wide enough to leave a typical
    bound-state basin unperturbed whilst still blocking transitions to
    neighbouring basins.
    """
    torsion_ids: Optional[list[tuple[int, int, int, int]]] = None
    """
    Explicit dihedrals to restrain, as ordered quartets of indices into the
    ligand. If defined, these override the automatic selection.

    Note: the same dihedrals must be restrained in the complex and solvent
    legs for the restraint contribution to cancel.
    """
    target_angles: Optional[list[AngleQuantity]] = None
    """
    Explicit target angles, one per entry in ``torsion_ids``. If ``None``,
    these are read from the ligand's input conformer. Can only be defined
    alongside ``torsion_ids``.
    """
    exclude_degenerate_rotors: bool = True
    """
    Whether to skip rotors whose rotation maps the molecule onto an equivalent
    structure, e.g. a -CF3 group or a monosubstituted phenyl. Such rotors may
    be trapped in one of their minima, but the minima are indistinguishable so
    there is no associated free energy error. Default True.
    """
    exclude_conjugated_carbonyls: bool = True
    """
    Whether to skip amide, ester and thioester bonds. Their barriers are far
    above thermal energy so they do not interconvert on simulation timescales.
    Default True.
    """

    @field_validator("spring_constant", "half_width")
    def positive_value(cls, v):
        if v.m < 0:
            errmsg = f"negative value passed: {v}"
            raise ValueError(errmsg)
        return v

    @field_validator("torsion_ids")
    def valid_torsion_ids(cls, v):
        if v is None:
            return v
        for quartet in v:
            if any(idx < 0 for idx in quartet):
                errmsg = "negative indices passed"
                raise ValueError(errmsg)
            if len(set(quartet)) != 4:
                errmsg = f"repeated atom index in torsion definition {quartet}"
                raise ValueError(errmsg)
        return v

    @model_validator(mode="after")
    def check_target_angles(self):
        if self.target_angles is None:
            return self
        if self.torsion_ids is None:
            errmsg = "target_angles can only be defined alongside torsion_ids"
            raise ValueError(errmsg)
        if len(self.target_angles) != len(self.torsion_ids):
            errmsg = (
                f"got {len(self.torsion_ids)} torsion_ids but "
                f"{len(self.target_angles)} target_angles, these must match"
            )
            raise ValueError(errmsg)
        return self
