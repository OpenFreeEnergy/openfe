# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Restraint geometry classes for intramolecular dihedral restraints.

These restraints are intended to hold a ligand's slow torsions in their
bound-state conformation whilst the ligand is alchemically decoupled, so that
the non-interacting end state does not sample conformers that the interacting
end state never visits.

Because a decoupled ligand does not interact with its environment, the free
energy of switching such a restraint on in the non-interacting end state is a
property of the isolated molecule alone. Applied identically in the complex and
solvent legs, the contribution therefore cancels in the resulting ddG and no
standard state correction is required. See
:class:`DihedralRestraintGeometry` for the conditions this relies on.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from pydantic import field_validator, model_validator

from .base import BaseRestraintGeometry


class DihedralRestraintGeometry(BaseRestraintGeometry):
    """
    A geometry class defining a set of dihedrals to restrain within a single
    molecule, alongside the angle each dihedral should be restrained to.

    Note
    ----
    The atom indices are indices into the full OpenMM System, and the order
    within each quartet matters: it defines the sign convention of the
    corresponding target angle.

    The cancellation of this restraint's free energy contribution between the
    complex and solvent legs requires that:

    * the non-interacting end state is genuinely non-interacting with its
      environment,
    * the identical geometry (atoms *and* target angles) is used in both legs,
    * the restraint is fully off in the interacting end state and fully on in
      the non-interacting end state of both legs,
    * no restrained dihedral is also restrained by the Boresch restraint on
      the same ligand (see :func:`validate_against_boresch_geometry`).
    """

    torsion_atoms: list[tuple[int, int, int, int]]
    """
    An ordered list of atom index quartets defining each restrained dihedral.
    """
    target_angles: list[float]
    """
    The target angle for each restrained dihedral, in radians, in the range
    (-pi, pi].
    """

    @field_validator("torsion_atoms")
    def positive_idxs(cls, v):
        if any(idx < 0 for quartet in v for idx in quartet):
            errmsg = "negative indices passed"
            raise ValueError(errmsg)
        return v

    @field_validator("torsion_atoms")
    def unique_idxs_per_torsion(cls, v):
        for quartet in v:
            if len(set(quartet)) != 4:
                errmsg = f"repeated atom index in torsion definition {quartet}"
                raise ValueError(errmsg)
        return v

    @model_validator(mode="after")
    def matching_lengths(self):
        if len(self.torsion_atoms) != len(self.target_angles):
            errmsg = (
                f"got {len(self.torsion_atoms)} torsions but "
                f"{len(self.target_angles)} target angles, these must match"
            )
            raise ValueError(errmsg)
        return self


def _heavy_neighbor_idxs(atom: Chem.Atom, exclude_idx: int) -> list[int]:
    """
    Get the indices of an atom's heavy (non-hydrogen) neighbors, excluding
    one given atom.
    """
    return [
        neighbor.GetIdx()
        for neighbor in atom.GetNeighbors()
        if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() != exclude_idx
    ]


def _is_conjugated_carbonyl_bond(bond: Chem.Bond) -> bool:
    """
    Whether a bond is the C-X bond of an amide, ester or thioester.

    These have barriers far above thermal energy and do not interconvert on
    simulation timescales, so they stay in whichever conformer the input pose
    defines and there is nothing to gain from restraining them.
    """
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    for heteroatom, carbon in ((begin, end), (end, begin)):
        if heteroatom.GetAtomicNum() not in (
        7, 8, 16) or carbon.GetAtomicNum() != 6:
            continue
        for carbon_bond in carbon.GetBonds():
            if carbon_bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if carbon_bond.GetOtherAtom(carbon).GetAtomicNum() in (8, 16):
                return True
    return False


def _is_rotatable_bond(
        bond: Chem.Bond,
        exclude_conjugated_carbonyls: bool = True,
) -> bool:
    """
    Whether a bond is a non-terminal, acyclic, single bond with at least one
    heavy substituent on each side, optionally excluding amide, ester and
    thioester bonds.
    """
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    if exclude_conjugated_carbonyls and _is_conjugated_carbonyl_bond(bond):
        return False

    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    if not _heavy_neighbor_idxs(begin, end.GetIdx()):
        return False
    if not _heavy_neighbor_idxs(end, begin.GetIdx()):
        return False
    return True


def _is_degenerate_rotor(
        bond: Chem.Bond,
        symmetry_classes: list[int],
) -> bool:
    """
    Whether rotating about a bond maps the molecule onto an equivalent
    structure, e.g. a -CF3, -C(CH3)3 or monosubstituted phenyl rotor.

    Such a rotor may well be trapped in one of its minima, but because the
    minima are indistinguishable there is no associated free energy error, so
    there is nothing to gain from restraining it.
    """
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    for atom, partner in ((begin, end), (end, begin)):
        substituents = [
            symmetry_classes[idx]
            for idx in _heavy_neighbor_idxs(atom, partner.GetIdx())
        ]
        if len(substituents) > 1 and len(set(substituents)) == 1:
            return True
    return False


def _select_torsion_atoms(
        bond: Chem.Bond,
        canonical_ranks: list[int],
) -> tuple[int, int, int, int]:
    """
    Pick a dihedral quartet spanning a rotatable bond.

    The outer atoms are chosen as the highest canonically ranked heavy
    neighbor on each side, so that the same quartet is picked every time the
    same molecule is passed in. This determinism matters: the complex and
    solvent legs must restrain identical dihedrals for the restraint
    contribution to cancel.
    """
    begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    outer_begin = max(
        _heavy_neighbor_idxs(bond.GetBeginAtom(), end_idx),
        key=lambda idx: canonical_ranks[idx],
    )
    outer_end = max(
        _heavy_neighbor_idxs(bond.GetEndAtom(), begin_idx),
        key=lambda idx: canonical_ranks[idx],
    )
    return (outer_begin, begin_idx, end_idx, outer_end)


def select_restrained_torsions(
        rdmol: Chem.Mol,
        exclude_degenerate_rotors: bool = True,
        exclude_conjugated_carbonyls: bool = True,
        excluded_bonds: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Heuristically select the dihedrals of a molecule worth restraining.

    Terminal rotors, ring bonds, amide bonds and rotors whose rotation is
    degenerate by symmetry are all excluded: either they sample well on
    simulation timescales, or their minima are indistinguishable, so
    restraining them adds restraint work without removing any sampling error.

    Parameters
    ----------
    rdmol : Chem.Mol
      The molecule to select torsions for. Indices in the returned quartets
      are indices into this molecule.
    exclude_degenerate_rotors : bool
      Whether to drop rotors related by symmetry, e.g. -CF3 or a
      monosubstituted phenyl. Default True.
    exclude_conjugated_carbonyls : bool
      Whether to drop amide, ester and thioester bonds. Default True.
    excluded_bonds : Optional[list[tuple[int, int]]]
      Central bonds to skip, given as pairs of indices into ``rdmol``. Use
      this to avoid restraining a torsion that a Boresch restraint on the
      same ligand already acts on.

    Returns
    -------
    list[tuple[int, int, int, int]]
      An ordered list of atom index quartets.
    """
    excluded = {frozenset(bond) for bond in (excluded_bonds or [])}
    # breakTies=False gives symmetry classes, breakTies=True gives a unique
    # deterministic ordering; we need both.
    symmetry_classes = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))
    canonical_ranks = list(Chem.CanonicalRankAtoms(rdmol, breakTies=True))

    torsions = []
    for bond in rdmol.GetBonds():
        if not _is_rotatable_bond(
                bond,
                exclude_conjugated_carbonyls=exclude_conjugated_carbonyls,
        ):
            continue
        if frozenset(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in excluded:
            continue
        if exclude_degenerate_rotors and _is_degenerate_rotor(bond,
                                                              symmetry_classes):
            continue
        torsions.append(_select_torsion_atoms(bond, canonical_ranks))
    return torsions


def get_dihedral_restraint_geometry(
        rdmol: Chem.Mol,
        ligand_idxs: list[int],
        torsion_atoms: list[tuple[int, int, int, int]] | None = None,
        target_angles: list[float] | None = None,
        exclude_degenerate_rotors: bool = True,
        exclude_conjugated_carbonyls: bool = True,
        excluded_bonds: list[tuple[int, int]] | None = None,
        conformer_id: int = 0,
) -> DihedralRestraintGeometry:
    """
    Get a DihedralRestraintGeometry for a ligand.

    Target angles are read from the ligand's input conformer, which is the
    bound pose, rather than from an equilibration trajectory. This keeps the
    restraint definition deterministic and identical between the complex and
    solvent legs without either leg having to depend on the other.

    Parameters
    ----------
    rdmol : Chem.Mol
      The ligand, with the conformer defining the bound pose.
    ligand_idxs : list[int]
      The indices of the ligand's atoms in the full OpenMM System, ordered to
      match ``rdmol``.
    torsion_atoms : Optional[list[tuple[int, int, int, int]]]
      Explicit dihedral quartets, as indices into ``rdmol``. If ``None``,
      these are selected heuristically.
    target_angles : Optional[list[float]]
      Explicit target angles in radians. If ``None``, these are read from the
      conformer.
    exclude_degenerate_rotors : bool
      See :func:`select_restrained_torsions`. Ignored if ``torsion_atoms`` is
      given.
    exclude_conjugated_carbonyls : bool
      See :func:`select_restrained_torsions`. Ignored if ``torsion_atoms`` is
      given.
    excluded_bonds : Optional[list[tuple[int, int]]]
      See :func:`select_restrained_torsions`. Ignored if ``torsion_atoms`` is
      given.
    conformer_id : int
      Which conformer of ``rdmol`` to read the target angles from.

    Returns
    -------
    DihedralRestraintGeometry
      An object defining the dihedral restraint geometry.
    """
    if torsion_atoms is None:
        torsion_atoms = select_restrained_torsions(
            rdmol=rdmol,
            exclude_degenerate_rotors=exclude_degenerate_rotors,
            exclude_conjugated_carbonyls=exclude_conjugated_carbonyls,
            excluded_bonds=excluded_bonds,
        )

    if target_angles is None:
        conformer = rdmol.GetConformer(conformer_id)
        target_angles = [
            rdMolTransforms.GetDihedralRad(conformer, *quartet)
            for quartet in torsion_atoms
        ]

    # wrap into (-pi, pi] so that serialised geometries compare cleanly
    target_angles = [
        float(angle - np.floor(angle / (2 * np.pi) + 0.5) * (2 * np.pi))
        for angle in target_angles
    ]

    return DihedralRestraintGeometry(
        torsion_atoms=[
            tuple(ligand_idxs[idx] for idx in quartet)  # type: ignore[misc]
            for quartet in torsion_atoms
        ],
        target_angles=target_angles,
    )


def validate_against_boresch_geometry(
        dihedral_geometry: DihedralRestraintGeometry,
        boresch_guest_atoms: list[int],
) -> None:
    """
    Check that no restrained dihedral shares a central bond with one of the
    dihedrals of a Boresch restraint on the same ligand.

    If it does, the Boresch restraint acts on the ligand's internal
    conformation and the decoupled ligand is no longer environment
    independent, which breaks the cancellation of the dihedral restraint
    contribution between the complex and solvent legs.

    Parameters
    ----------
    dihedral_geometry : DihedralRestraintGeometry
      The dihedral restraint geometry to check.
    boresch_guest_atoms : list[int]
      The ordered guest atoms (L1, L2, L3) of the Boresch restraint, as
      indices into the full OpenMM System.

    Raises
    ------
    ValueError
      If a restrained dihedral shares a central bond with the Boresch
      restraint's phi_B or phi_C dihedrals.
    """
    guest_bonds = {
        frozenset((boresch_guest_atoms[0], boresch_guest_atoms[1])),
        frozenset((boresch_guest_atoms[1], boresch_guest_atoms[2])),
    }
    for quartet in dihedral_geometry.torsion_atoms:
        central_bond = frozenset((quartet[1], quartet[2]))
        if central_bond in guest_bonds:
            errmsg = (
                f"The restrained dihedral {quartet} shares its central bond "
                f"with the Boresch restraint guest atoms {boresch_guest_atoms}. "
                "The Boresch restraint would then act on the ligand's internal "
                "conformation in the complex leg but not in the solvent leg, so "
                "the dihedral restraint contribution would no longer cancel "
                "between legs. Pick Boresch guest anchors within a rigid "
                "fragment, or exclude this bond from the dihedral restraints."
            )
            raise ValueError(errmsg)