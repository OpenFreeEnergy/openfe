from dataclasses import dataclass, field, fields
from gufe import LigandAtomMapping
from openff.toolkit import ForceField
from collections import defaultdict
from rdkit import Chem
from typing import Any, Iterable
from shutil import copy
import pathlib

@dataclass
class JunctionData:
    junction_atom: int
    dummies: set[int] = field(default_factory=set)
    physical: set[int] = field(default_factory=set)


@dataclass
class CorrectionData:
    removed_angles: set[frozenset[int]] = field(default_factory=set)
    softened_angles: set[frozenset[int]] = field(default_factory=set)
    stiffened_angles: set[frozenset[int]] = field(default_factory=set)
    removed_dihedrals: set[frozenset[int]] = field(default_factory=set)
    stiffened_dihedrals: set[frozenset[int]] = field(default_factory=set)
    removed_impropers: set[frozenset[int]] = field(default_factory=set)

    def merge(self, other: "CorrectionData") -> "CorrectionData":
        """Merge another CorrectionData object into this one."""
        for f in fields(self):
            current_set = getattr(self, f.name)
            other_set = getattr(other, f.name)
            current_set.update(other_set)
        return self


def _remap_corrections_to_system_indices(
    corrections: dict[str, "CorrectionData"],
    old_mol_offset: int,
    new_mol_offset: int,
) -> dict[str, "CorrectionData"]:
    """
    Remap atom indices in CorrectionData from isolated-molecule indices to
    full-system (solvated / protein) indices.

    _derive_dummy_junction_corrections produces corrections whose atom indices
    are 0-based within each isolated molecule.  When the systems are solvated
    or combined with a protein the ligand atoms are no longer at index 0 in the
    topology, so the corrections must be shifted before they are consumed by the
    HTF.

    Parameters
    ----------
    corrections : dict[str, CorrectionData]
        Corrections keyed by "lambda_0" and "lambda_1" with molecule-level indices.
    old_mol_offset : int
        Index of the first atom of componentA in the stateA system topology.
        Use ``ligand_mappings['old_mol_indices'][0]`` from ``get_system_mappings``.
    new_mol_offset : int
        Index of the first atom of componentB in the stateB system topology.
        Use ``ligand_mappings['new_mol_indices'][0]`` from ``get_system_mappings``.

    Returns
    -------
    dict[str, CorrectionData]
        New corrections dict with system-level atom indices.
    """

    def _shift(correction: CorrectionData, offset: int) -> CorrectionData:
        remapped = CorrectionData()
        for f in fields(correction):
            shifted = {
                frozenset(idx + offset for idx in atom_set)
                for atom_set in getattr(correction, f.name)
            }
            setattr(remapped, f.name, shifted)
        return remapped

    return {
        # lambda_1 corrections reference componentA (old/stateA system) atoms
        "lambda_1": _shift(corrections["lambda_1"], old_mol_offset),
        # lambda_0 corrections reference componentB (new/stateB system) atoms
        "lambda_0": _shift(corrections["lambda_0"], new_mol_offset),
    }

def _derive_dummy_junction_corrections(
    mapping: LigandAtomMapping, force_field: str
) -> dict[str, CorrectionData]:
    """
    For the given mapping and SMIRNOFF style forcefield, derive dummy junction corrections based on the best practices of Fleck et al.

    Parameters
    ----------
    mapping : LigandAtomMapping
        The atom mapping between the two ligands, used to determine the core and dummy atoms of each end state.
    force_field : str
        The name of the SMIRNOFF style force field to use to derive the corrections, e.g. "openff-2.3.0.offxml".

    TODO
    - check if torsion ends in a collinear group or a group which becomes collinear

    Notes
    -----
    - Currently only works for SMIRNOFF force fields and should be extended in future
    - Uses the single bond, angle, dihedral approach for each junction type
    - Improper torsions spanning the core-dummy junctions are only keep if they involve the chosen anchor atoms and dummies.
    """
    all_corrections = {"lambda_0": CorrectionData(), "lambda_1": CorrectionData()}
    # load the force field to label the molecules with parameters
    if not force_field.endswith(".offxml"):
        force_field = force_field + ".offxml"
    ff = ForceField(force_field)

    # loop over the end states and get the corrections for each junction
    for state_key in ["lambda_0", "lambda_1"]:
        print(f"Finding corrections for {state_key}")
        # the corrections for the end state are derived from the ligand at the opposite end state
        if state_key == "lambda_0":
            smc = mapping.componentB
            core_atoms = set(mapping.componentA_to_componentB.values())
        else:
            smc = mapping.componentA
            core_atoms = set(mapping.componentA_to_componentB.keys())
        print(f"Ligand at this state is: {smc} with core atoms {core_atoms}")
        rdkit_mol = smc.to_openff().to_rdkit()

        # dummy atoms are all non-mapped atoms
        dummy_atoms = {
            a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetIdx() not in core_atoms
        }
        # label the molecule with parameters
        ff_labels = ff.label_molecules(smc.to_openff().to_topology())[0]
        # tag any free rotors in the molecule as we may need to stiffen torsions which end in the rotor group
        rotor_atoms = _find_free_rotors(rdkit_mol)
        print(f"Found rotor atoms: {rotor_atoms}")
        # find the dummy junctions
        dummy_junctions = _find_dummy_junctions_rdkit(rdkit_mol, dummy_atoms)

        # now derive the corrections for each junction
        for junction in dummy_junctions:
            # dispatch the junction to the correction
            physicals = len(junction.physical)
            match physicals:
                case 1:
                    print(
                        f"Deriving corrections for terminal junction with physical atom {junction.physical} and dummy atoms {junction.dummies}"
                    )
                    corrections = _derive_terminal_corrections(
                        junction, rdkit_mol, ff_labels, core_atoms, rotor_atoms
                    )
                    print(f"Derived corrections for terminal junction: {corrections}")
                case 2:
                    print(
                        f"Deriving corrections for dual junction with physical atoms {junction.physical} and dummy atoms {junction.dummies}"
                    )
                    corrections = _derive_dual_corrections(
                        junction, rdkit_mol, ff_labels, core_atoms, rotor_atoms
                    )
                    print(f"Derived corrections for dual junction: {corrections}")
                case 3:
                    print(
                        f"Deriving corrections for triple junction with physical atoms {junction.physical} and dummy atoms {junction.dummies}"
                    )
                    corrections = _derive_triple_corrections(
                        junction, rdkit_mol, ff_labels, core_atoms
                    )
                    print(f"Derived corrections for triple junctions: {corrections}")
                case _:
                    raise NotImplementedError(
                        f"Higher order junctions not currently supported junction is order {physicals}."
                    )

            # validate the corrections
            if not _check_dummy_junction(
                junction=junction,
                corrections=corrections,
                force_field_labels=ff_labels,
                core_atoms=core_atoms,
            ):
                raise ValueError(
                    f"Corrections failed validation for junction with physical atoms {junction.physical} and dummy atoms {junction.dummies}. Corrections were: {corrections}"
                )
            # add these corrections to the overall corrections dict
            all_corrections[state_key].merge(corrections)

            # remove improper torsions which couple the dummy and core atoms unless they end in the core junction atom
            improper_corrections = _find_improper_corrections(junction, ff_labels)
            all_corrections[state_key].merge(improper_corrections)
    return all_corrections


def _find_dummy_junctions_rdkit(
    rdkit_mol: Chem.Mol, dummy_atoms: set[int]
) -> list[JunctionData]:
    """Identify dummy-core junctions in the given molecule for the set of dummy atoms extracted from an atom mapping.

    Parameters
    ----------
    rdkit_mol : Chem.Mol
        The RDKit molecule object to find the dummy-core junctions for.
    dummy_atoms: set[int]
        The set of atom indices to be considered dummy atoms in the molecule (they are not mapped).

    Returns
    -------
    list[JunctionData]
        A list of junction data dictionaries, each containing the junction atom index, the set of dummy atom indices, and the set of physical atom indices bonded to the junction atom.


    Notes
    -----
    Dummy junctions involving a physical and dummy atom in the same ring system are excluded as we can not correctly
    handle ring openings.
    """
    # make a lookup for the bonds
    bond_look_up = defaultdict(list)
    for bond in rdkit_mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_look_up[a1].append(a2)
        bond_look_up[a2].append(a1)

    # get the ring info
    atom_rings = rdkit_mol.GetRingInfo().AtomRings()

    assigned_dummies = set()
    dummy_junctions = []
    for dummy_atom in dummy_atoms:
        if dummy_atom in assigned_dummies:
            continue
        bonded_physicals = [a for a in bond_look_up[dummy_atom] if a not in dummy_atoms]
        # if there are no bonded physical atoms this is part of a larger dummy group so skip
        if len(bonded_physicals) == 0:
            continue
        junction_atom = bonded_physicals[0]

        # if the junction atom and the dummy atom are in the same ring skip this dummy atom
        if any(junction_atom in ring and dummy_atom in ring for ring in atom_rings):
            continue

        other_dummies = [
            a
            for a in bond_look_up[junction_atom]
            if a in dummy_atoms and a != dummy_atom
        ]
        dummy_junctions.append(
            JunctionData(
                junction_atom=junction_atom,
                dummies={dummy_atom, *other_dummies},
                physical={
                    a for a in bond_look_up[junction_atom] if a not in dummy_atoms
                },
            )
        )
        assigned_dummies.update([dummy_atom] + other_dummies)
    return dummy_junctions


def _find_free_rotors(rdkit_mol: Chem.Mol) -> set[int]:
    """
    Get the atom indices of any free rotor terminal atoms in the molecule.

    Parameters
    ----------
    rdkit_mol : Chem.Mol
        The RDKit molecule object to search for free rotors.

    Returns
    -------
    set[int]
        A set of atom indices corresponding to free rotor terminal atoms.

    Note
    ----
    We define free rotors as any terminal atoms in the following groups:
        - CH3
        - NH2
        - OH
        - SH
    """
    rotor_indices = set()
    # smarts wrote to match the terminal atom and the connected atom
    for rotor_smarts in ["[H]-[CX4H3]", "[H]-[NX3H2]", "[H]-[OX2H]", "[H]-[SX2H]"]:
        rotor_query = Chem.MolFromSmarts(rotor_smarts)
        matches = rdkit_mol.GetSubstructMatches(rotor_query)
        # get the terminal atom in each match and add to the set
        for match in matches:
            terminal_atoms = [
                a for a in match if rdkit_mol.GetAtomWithIdx(a).GetDegree() == 1
            ]
            rotor_indices.update(terminal_atoms)
    return rotor_indices


def _get_heaviest_dihedral_anchor(
    dihedrals: Iterable[tuple[int, int, int, int]],
    excluded_atoms: set[int],
    rdkit_mol: Chem.Mol,
    junction_atom: int,
) -> int:
    """
    Get the heaviest terminal atom from a list of dihedrals, excluding any atoms in the excluded_atoms set.

    Parameters
    ----------
    dihedrals : Iterable[tuple[int, int, int, int]]
        An iterable of dihedral tuples which span the dummy-core junction.
    excluded_atoms : set[int]
        A set of atoms which should not be considered when finding the heaviest dihedral anchor.
    rdkit_mol : Chem.Mol
        The RDKit molecule object to use for atom mass lookups.
    junction_atom : int
        The index of the junction atom in the molecule.

    Returns
    -------
    int
        The heaviest dihedral anchor index from the list of dihedrals.

    Notes
    -----
    - This is a deterministic method of finding the anchor atoms to ensure cancellation between repeats
    - If the physical junction atom is planer and in a ring we target heavy atoms in the same ring to help with stiffened dihedrals
    """
    terminal_atoms_with_mass = []
    for dihedral in dihedrals:
        terminal_atom = [a for a in dihedral if a not in excluded_atoms][0]
        mass = rdkit_mol.GetAtomWithIdx(terminal_atom).GetMass()
        terminal_atoms_with_mass.append((terminal_atom, mass))

    # filter the terminal atoms list to end in a ring if the junction atom is in a ring
    # this helps with stiffened dihedrals
    if (
        len(terminal_atoms_with_mass) > 1
        and rdkit_mol.GetAtomWithIdx(junction_atom).IsInRing()
    ):
        junction_rings = [
            ring
            for ring in rdkit_mol.GetRingInfo().AtomRings()
            if junction_atom in ring
        ]
        terminal_atoms_with_mass = [
            term_atom
            for term_atom in terminal_atoms_with_mass
            if any(term_atom[0] in ring for ring in junction_rings)
        ]

    # get the atom with the largest mass second sort by index
    return max(terminal_atoms_with_mass, key=lambda x: (x[1], x[0]))[0]


def _derive_terminal_corrections(
    junction: JunctionData,
    rdkit_mol: Chem.Mol,
    force_field_labels: dict[str, Any],
    core_atoms: set[int],
    free_rotors: set[int],
) -> CorrectionData:
    """
    Derive the corrections for a terminal dummy junction as per the best practices of Fleck et al.

    Parameters
    ----------
    junction : JunctionData
        The junction data dictionary containing the junction atom index, the set of dummy atom indices, and the set of physical atom indices bonded to the junction atom.
    rdkit_mol : Chem.Mol
        The RDKit molecule object to derive the corrections for.
    force_field_labels : dict[str, Any]
        The force field labels for the molecule, used to identify the relevant terms to keep and remove.
    core_atoms : set[int]
        The set of core atom indices in the molecule.
    free_rotors : set[int]
        The set of free rotor terminal atom indices in the molecule.

    Returns
    -------
    CorrectionData
        A correctionData object containing the corrections for the terminal dummy junction considered.

    Notes
    -----
    - This results in a single bond, angle and dihedral term being kept per terminal junction
    - All dihedrals will terminate in a single heavy atom to avoid single and dual anchor dihedral constraints
    - If the dihedral terminates in a free rotor we stiffen the dihedral and change the periodicity to 1 and the phase to 0 to
        try and uncouple the dummy from the rapid rotations of the free rotor.
    - In the case this is a dummy group with a terminal junction we do not do any dihedral removal as all dihedrals should end
     in the same physical atom due to the geometry of the terminal junction.
    """
    junction_atom = junction.junction_atom
    dummy_atoms = junction.dummies
    physical_atoms = junction.physical

    # track those which terminate in the dummy junction atom (bonded to the physical junction atom)
    dummy_junction_dihedrals = set()
    other_core_atoms = core_atoms - physical_atoms - {junction_atom}
    for dihedral in force_field_labels["ProperTorsions"].keys():
        # do checks to classify the dihedral
        has_junction = junction_atom in dihedral
        has_dummy = dummy_atoms.intersection(dihedral)
        has_physical = physical_atoms.intersection(dihedral)
        has_other_core = other_core_atoms.intersection(dihedral)

        # check if the dihedral terminates at the dummy junction atom and involves no other dummy atoms
        if has_junction and has_dummy and has_physical and has_other_core:
            dummy_junction_dihedrals.add(dihedral)

    # now we need to determine which dihedral to keep,
    # get the set of atoms to ignore when finding the terminal atom
    excluded_atoms = set(dummy_atoms) | set(physical_atoms) | {junction_atom}
    heaviest_terminal_atom = _get_heaviest_dihedral_anchor(
        dihedrals=dummy_junction_dihedrals,
        excluded_atoms=excluded_atoms,
        rdkit_mol=rdkit_mol,
        junction_atom=junction_atom,
    )
    corrections = CorrectionData()
    # remove all dihedrals which do not terminate in the heaviest terminal atom
    for dihedral in dummy_junction_dihedrals:
        if heaviest_terminal_atom not in dihedral:
            corrections.removed_dihedrals.add(frozenset(dihedral))
        elif (
            heaviest_terminal_atom in dihedral and heaviest_terminal_atom in free_rotors
        ):
            # if the dihedral terminates in a free rotor we stiffen the dihedral to
            # try and uncouple the dummy from the rapid rotations of the free rotor.
            corrections.stiffened_dihedrals.add(frozenset(dihedral))

    # check that we don't have multiple path dihedrals left in the system which can happen in 4 membered rings
    corrections = _prune_multiple_path_dihedrals(
        target_dihedrals=dummy_junction_dihedrals, corrections=corrections
    )
    return corrections


def _derive_dual_corrections(
    junction: JunctionData,
    rdkit_mol: Chem.Mol,
    force_field_labels: dict[str, Any],
    core_atoms: set[int],
    rotor_atoms: set[int],
) -> CorrectionData:
    """
    Derive the corrections for a dual dummy junction as per the best practices of Fleck et al.

    Notes
    -----
    - In the case of a dual junction with two dummy groups we do not remove any valence terms between them as we use the
        single bond-angle-dihedral method to avoid the groups overlapping.
    - If the physical junction atom is planar we stiffen the dihedral we retain to prevent the junction flapping, we use a k value
        of 100 kcal/mol a periodicity of 1 and a phase of PI to set the equilibrium value to 0.
    """
    junction_atom = junction.junction_atom
    dummy_atoms = junction.dummies
    physical_atoms = junction.physical
    # track those which terminate in the dummy junction atom (bonded to the physical junction atom)
    dummy_junction_dihedrals = set()
    # also track dihedrals which terminate at one of the physical atoms and originate from within one of the dummy groups
    dummy_group_dihedrals = set()
    other_core_atoms = core_atoms - physical_atoms - {junction_atom}

    for dihedral in force_field_labels["ProperTorsions"].keys():
        # do checks to classify the dihedral
        has_junction = junction_atom in dihedral
        has_dummy = dummy_atoms.intersection(dihedral)
        has_physical = physical_atoms.intersection(dihedral)
        has_other_core = other_core_atoms.intersection(dihedral)

        # check if the dihedral terminates at the dummy junction atom and involves no other dummy atoms
        if has_junction and has_dummy and has_physical and has_other_core:
            dummy_junction_dihedrals.add(dihedral)
        # check if the dihedral originates from the dummy group but not the dummy junction atom
        elif has_junction and has_dummy and has_physical and not has_other_core:
            # this could also be a dihedral linking two dummy groups - this requires the central atoms to be the core junction and a physical atom
            central_atoms = {dihedral[1], dihedral[2]}
            if junction_atom in central_atoms and central_atoms.intersection(
                physical_atoms
            ):
                continue
            dummy_group_dihedrals.add(dihedral)

    corrections = CorrectionData()

    # find the dihedral to terminate in by mass
    excluded_atoms = set(dummy_atoms) | set(physical_atoms) | {junction_atom}
    heaviest_terminal_atom = _get_heaviest_dihedral_anchor(
        dihedrals=dummy_junction_dihedrals,
        excluded_atoms=excluded_atoms,
        rdkit_mol=rdkit_mol,
        junction_atom=junction_atom,
    )

    # we need to check if the junction is planar and if we need to stiffen the dihedral
    junction_hybridisation = rdkit_mol.GetAtomWithIdx(junction_atom).GetHybridization()
    if junction_hybridisation == Chem.HybridizationType.SP2:
        junction_is_planar = True
    else:
        junction_is_planar = False

    # remove all dihedrals which do not terminate in the heaviest terminal atom
    for dihedral in dummy_junction_dihedrals:
        if heaviest_terminal_atom not in dihedral:
            corrections.removed_dihedrals.add(frozenset(dihedral))
        # we need to stiffen the dihedral if we want to keep the dummy group in plane with the rest of the molecule
        # or if we have to anchor using a free rotor
        # elif heaviest_terminal_atom in dihedral and (
        #     heaviest_terminal_atom in rotor_atoms
        # ):
        elif heaviest_terminal_atom in dihedral and (heaviest_terminal_atom in rotor_atoms or junction_is_planar): # disable dihedral stiffening for now
            corrections.stiffened_dihedrals.add(frozenset(dihedral))

    # check that we don't have multiple path dihedrals left in the system which can happen in 4 membered rings
    corrections = _prune_multiple_path_dihedrals(
        target_dihedrals=dummy_junction_dihedrals, corrections=corrections
    )

    # find the physical atom the kept dihedrals pass through
    heavy_atom = rdkit_mol.GetAtomWithIdx(heaviest_terminal_atom)
    physical_atom_to_keep = [
        a.GetIdx() for a in heavy_atom.GetNeighbors() if a.GetIdx() in physical_atoms
    ][0]

    # remove redundant angles not running through the kept dihedral to minimise the number of phsycial anchor atoms
    for angle in force_field_labels["Angles"].keys():
        if (
            dummy_atoms.intersection(angle)
            and junction_atom in angle
            and physical_atoms.intersection(angle)
            and physical_atom_to_keep not in angle
        ):
            # if the angle involves the junction atom, a dummy atom and a physical atom it might be redundant
            # dual junctions have two possible branches
            corrections.removed_angles.add(frozenset(angle))

    # remove single and dual anchor dihedral constraints if the dummy group is larger than a single atom
    if dummy_group_dihedrals:
        print("found dummy group dihedrals", dummy_group_dihedrals)
        # only keep torsions which terminate in the physical atom kept in the dummy junction dihedral
        for dihedral in dummy_group_dihedrals:
            if physical_atom_to_keep not in dihedral:
                corrections.removed_dihedrals.add(frozenset(dihedral))

    return corrections


def _derive_triple_corrections(
    junction: JunctionData, rdkit_mol, force_field_labels, core_atoms
) -> CorrectionData:
    """
    Derive the corrections for a triple dummy junction as per the best practices of Fleck et al.

    Notes
    -----
    - Non-planar triple junctions can not be fully separated, so we just soften all angles and remove dihedrals terminating in the
        dummy junction atom originating from the physical system.
    - Non-planar triple junctions with large dummy groups will have dihedrals removed to avoid single and dual anchor dihedral constraints.
    """
    corrections = CorrectionData()

    junction_atom = junction.junction_atom
    physical_atoms = junction.physical
    dummy_atoms = junction.dummies

    # determine the nature of the junction by the hybridisation of the physical junction atom
    junction_hybridisation = rdkit_mol.GetAtomWithIdx(junction_atom).GetHybridization()

    if junction_hybridisation == Chem.HybridizationType.SP2:
        # is this type possible with our hybrid topology setup?
        print("Junction is planar, applying planar junction corrections.")
        raise NotImplementedError("Planar triple junctions not supported")
    else:
        print("Junction is non-planar, applying non-planar junction corrections.")
        # first we need to find the three angles to soften
        for angle in force_field_labels["Angles"].keys():
            has_junction_a = junction_atom in angle
            # we only want a single dummy atom in the angle
            has_dummy_a = len(dummy_atoms.intersection(angle)) == 1
            has_physical_a = len(physical_atoms.intersection(angle)) == 1
            if has_junction_a and has_dummy_a and has_physical_a:
                corrections.softened_angles.add(frozenset(angle))

        # make sure we found 3 angles to soften as expected
        if len(corrections.softened_angles) != 3:
            raise ValueError(
                f"Expected to find 3 angles to soften for a triple junction, but found {len(corrections.softened_angles)}. "
                f"Found the following softened angles: {corrections.softened_angles}"
            )

        # remove all dihedrals which terminate in the dummy junction atom and originate from the physical system
        # also catch any dihedrals which terminate in a physical atom and originate from within the dummy group
        other_core_atoms = (
            core_atoms
            - physical_atoms
            - {
                junction_atom,
            }
        )
        dummy_group_dihedrals = set()
        for dihedral in force_field_labels["ProperTorsions"].keys():
            # classify the dihedral type
            has_junction = junction_atom in dihedral
            has_dummy = dummy_atoms.intersection(dihedral)
            has_physical = physical_atoms.intersection(dihedral)
            has_other_core = other_core_atoms.intersection(dihedral)

            # if the dihedral originates from the physical and terminates in the dummy junction we need to remove
            # len(has_physical) == 2 is a special case where the tiple junction is on a cyclopropane group
            if has_dummy and has_physical and has_junction and (has_other_core or len(has_physical) == 2):
                corrections.removed_dihedrals.add(frozenset(dihedral))
            # if the dihedral originates from the dummy group and terminates in the physical atom we need to collect for
            # anchor corrections
            # check if the dihedral originates from the dummy group but not the dummy junction atom
            elif has_junction and has_dummy and has_physical and not has_other_core:
                # this could also be a dihedral linking two dummy groups - this requires the central atoms to be the core junction and a physical atom
                central_atoms = {dihedral[1], dihedral[2]}
                if junction_atom in central_atoms and central_atoms.intersection(
                    physical_atoms
                ):
                    continue
                dummy_group_dihedrals.add(dihedral)

        if dummy_group_dihedrals:
            # find the heaviest terminal atom to anchor dihedrals from the dummy group to
            # we expect this to be one of the physical atoms
            all_dummies = {
                a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetIdx() not in core_atoms
            }
            excluded_atoms = all_dummies | {junction_atom}
            dummy_group_terminal_atom = _get_heaviest_dihedral_anchor(
                dihedrals=dummy_group_dihedrals,
                excluded_atoms=excluded_atoms,
                rdkit_mol=rdkit_mol,
                junction_atom=junction_atom,
            )

            # remove any dihedrals which do not terminate in this core atom
            for dihedral in dummy_group_dihedrals:
                if dummy_group_terminal_atom not in dihedral:
                    corrections.removed_dihedrals.add(frozenset(dihedral))

    return corrections


def _find_improper_corrections(
    junction: JunctionData, force_field_labels: dict[str, Any]
) -> CorrectionData:
    """
    Find any improper torsions which need to be removed for a given junction.

    Notes
    -----
    - We remove any improper torsions which couple dummy and core atoms beyond the junction atom, so impropers terminating
    in the core junction atom are allowed.
    """
    corrections = CorrectionData()
    junction_atom = junction.junction_atom
    dummy_atoms = junction.dummies
    physical_atoms = junction.physical
    # remove the junction atom from the core atoms as we want improper torsions only covering dummy and this atom
    for dihedral in force_field_labels["ImproperTorsions"].keys():
        # if the improper crosses the core dummy junction and involves a physical atom remove it
        if (
            physical_atoms.intersection(dihedral)
            and junction_atom in dihedral
            and dummy_atoms.intersection(dihedral)
        ):
            corrections.removed_impropers.add(frozenset(dihedral))
    return corrections


def _prune_multiple_path_dihedrals(
    target_dihedrals: set[tuple[int, int, int, int]], corrections: CorrectionData
) -> CorrectionData:
    r"""
    For the given set of target dihedrals which have been processed prune any left over redundant dihedrals.

    Notes
    -----
    Consider the following dual junction, if we are to keep a single bond, angle and dihedral term per dummy atom (D1&2)
    then there are two equivalent dihedrals which terminate in the same heavy physical atom (C4) so we then need to prune them
    to a single one in a deterministic way.

       D1     C2
         \  /    \
          C1      C4 -
        /  \    /
     D2      C3

    Only the lowest sum of the dihedral index is retained when multiple dihedrals are available.
    """
    dihedrals_by_terminals = defaultdict(set)

    for dihedral in target_dihedrals:
        dihedral_fs = frozenset(dihedral)
        if (
            dihedral_fs not in corrections.removed_dihedrals
            and dihedral_fs not in corrections.stiffened_dihedrals
        ):
            dihedrals_by_terminals[frozenset([dihedral[0], dihedral[3]])].add(dihedral)

    for terminal_atoms, dihedrals in dihedrals_by_terminals.items():
        if len(dihedrals) > 1:
            # we have multiple dihedrals with the same terminal atoms, we need to prune to a single one
            dihedral_to_keep = min(dihedrals, key=lambda x: sum(x))
            for dihedral in dihedrals:
                if dihedral != dihedral_to_keep:
                    corrections.removed_dihedrals.add(frozenset(dihedral))
    return corrections


def _check_dummy_junction(
    junction: JunctionData,
    corrections: CorrectionData,
    force_field_labels: dict[str, Any],
    core_atoms: set[int],
) -> bool:
    """
    Check the dummy junction atom is correctly connected via only 3 or less non-redundant valence terms.
    """
    junction_atom = junction.junction_atom
    dummy_atoms = junction.dummies
    physical_atoms = junction.physical
    other_core_atoms = core_atoms - physical_atoms - {junction_atom}
    # check there is a single bond/constraint between the dummy and physical junction atoms
    for dummy_atom in dummy_atoms:
        valence_terms = {
            "bonds": set(),
            "angles": set(),
            "dihedrals": set(),
        }
        # check bond terms
        junction_bond = {dummy_atom, junction_atom}
        for bond in force_field_labels["Bonds"].keys():
            if set(bond) == junction_bond:
                valence_terms["bonds"].add(bond)
        if len(valence_terms["bonds"]) != 1:
            return False

        # check angle terms
        for angle in force_field_labels["Angles"].keys():
            angle = frozenset(angle)
            junction_in_angle = junction_atom in angle
            dummy_in_angle = dummy_atom in angle
            physical_in_angle = len(physical_atoms.intersection(angle)) == 1
            if junction_in_angle and dummy_in_angle and physical_in_angle:
                if angle not in corrections.removed_angles:
                    valence_terms["angles"].add(angle)

        # now check dihedral terms
        for dihedral in force_field_labels["ProperTorsions"].keys():
            dihedral = frozenset(dihedral)
            junction_in_dihedral = junction_atom in dihedral
            dummy_in_dihedral = dummy_atom in dihedral
            physical_in_dihedral = physical_atoms.intersection(dihedral)
            other_core_in_dihedral = other_core_atoms.intersection(dihedral)
            if (
                junction_in_dihedral
                and dummy_in_dihedral
                and physical_in_dihedral
                and (other_core_in_dihedral or len(physical_in_dihedral) == 2)
            ):
                if dihedral not in corrections.removed_dihedrals:
                    valence_terms["dihedrals"].add(dihedral)

        # now we need to check the valence terms are correct
        total_terms = (
            len(valence_terms["bonds"])
            + len(valence_terms["angles"])
            + len(valence_terms["dihedrals"])
        )
        # bond (1) + up to 3 softened angles is 4 valid connections
        if total_terms == 4 and all(
            angle in corrections.softened_angles for angle in valence_terms["angles"]
        ):
            continue
        if total_terms > 3 or total_terms == 0:
            return False

    return True


def _draw_dummy_corrections(
    mapping: LigandAtomMapping,
    corrections: dict[str, CorrectionData],
    output_dir: pathlib.Path,
    force_field: str,
) -> None:
    """
    Draw the dummy junction corrections for each end state in the mapping and save to the output directory.

    Parameters
    ----------
    mapping : LigandAtomMapping
        The ligand atom mapping containing the molecules to draw.
    corrections : dict[str, CorrectionData]
        The corrections to draw for each state, as derived from _derive_dummy_junction_corrections.
    output_dir : pathlib.Path
        The directory to save the drawn corrections to these will be named {mapping.componentA.name}_corrections.svg and {mapping.componentB.name}_corrections.svg for lambda = 1 and lambda = 0 respectively.
    force_field : str
        The SMIRNOFF force field which should be used to label the molecules to get the initial parameters.

    Notes
    -----
    The atom highlighting colours are:
    - Light green: core atoms
    - Light grey: dummy atoms
    The valence term highlighting colours are:
    - Green: normal term crossing the junction
    - Red: stiffened term crossing the junction
    - Blue: softened term crossing the junction

    The output directory will be created if it does not exist
    """
    from rdkit.Chem import AllChem, Draw

    # define some atom colours
    core_atom_colour = (0.7, 1.0, 0.7)  # light green
    dummy_atom_colour = (0.5, 0.5, 0.5)  # light grey
    normal_term_colour = (0.0, 1.0, 0.0)  # green
    stiffened_term_colour = (1.0, 0.0, 0.0)  # red
    softened_term_colour = (0.0, 0.0, 1.0)  # blue

    ff = ForceField(force_field)

    # the corrections for each ligand are stored in the opposite end state
    for ligand, state_index in zip(
        [mapping.componentA, mapping.componentB], ["lambda_1", "lambda_0"]
    ):
        correction_data = corrections[state_index]
        rdkit_mol = ligand.to_openff().to_rdkit()
        # get a 2D conformer for drawing
        AllChem.Compute2DCoords(rdkit_mol)

        # get the core atoms to highlight
        core_atoms = set(
            mapping.componentA_to_componentB.keys()
            if state_index == "lambda_1"
            else mapping.componentA_to_componentB.values()
        )
        dummy_atoms = set(
            [a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetIdx() not in core_atoms]
        )

        # get the labels of the molecule to find the parameters for the valence terms
        ff_labels = ff.label_molecules(ligand.to_openff().to_topology())[0]

        # find the junctions of this end state
        print("finding junctions for ", ligand.name)
        junctions = _find_dummy_junctions_rdkit(
            rdkit_mol=rdkit_mol, dummy_atoms=dummy_atoms
        )

        # we want to group the terms by junction to make the drawing clearer
        to_draw_by_junction = defaultdict(list)

        for junction in junctions:
            junction_atom = junction.junction_atom
            junction_dummies = junction.dummies

            # search through the valence labels bonds, angles, dihedrals and impropers to find any terms which pass
            # through the junction
            for bond in ff_labels["Bonds"].keys():
                bond_set = frozenset(bond)
                if junction_atom in bond_set and any(
                    dummy in bond_set for dummy in junction_dummies
                ):
                    # this is the bond crossing the junction
                    # find the index of the bond in the rdkit molecule
                    bond_idx = rdkit_mol.GetBondBetweenAtoms(bond[0], bond[1]).GetIdx()

                    # check if this could be a constraint, by checking for a H
                    if any(
                        rdkit_mol.GetAtomWithIdx(a).GetSymbol() == "H" for a in bond
                    ):
                        label = "Constraint"
                    else:
                        label = "Bond"

                    # add this to the list of terms to draw for this junction
                    to_draw_by_junction[bond_set].append(
                        {
                            "legend": label + " " + repr([int(x) for x in bond]),
                            "Bonds": [bond_idx],
                            "Bond Colours": {bond_idx: normal_term_colour},
                            "Atoms": bond,
                        }
                    )

            for angle in ff_labels["Angles"].keys():
                angle_set = frozenset(angle)
                if junction_atom in angle_set and any(
                    dummy in angle_set for dummy in junction_dummies
                ):
                    # this is an angle crossing the junction
                    # check if it has been removed, if removed just skip
                    if angle_set in correction_data.removed_angles:
                        continue

                    # if this angle has been kept we need to draw it
                    p1, p2, p3 = angle
                    # find the bonds for the angle to highlight
                    bond1_idx = rdkit_mol.GetBondBetweenAtoms(p1, p2).GetIdx()
                    bond2_idx = rdkit_mol.GetBondBetweenAtoms(p2, p3).GetIdx()
                    # find the junction bond this should be stored under
                    junction_bond = [
                        atom
                        for atom in angle
                        if atom == junction_atom or atom in junction_dummies
                    ]
                    # this should always be 2 atoms unless it is a dummy only angle which spans a core atom
                    if len(junction_bond) == 3:
                        # this will always include the junction atom in this case as its the central atom of the bond
                        junction_bond = junction_bond[:2]

                    junction_bond = frozenset(junction_bond)

                    # work out the colour and label for this angle
                    if angle_set in correction_data.softened_angles:
                        legend = "Softened Angle"
                        bond_colour = softened_term_colour
                    elif angle_set in correction_data.stiffened_angles:
                        legend = "Stiffened Angle"
                        bond_colour = stiffened_term_colour
                    else:
                        legend = "Angle"
                        bond_colour = normal_term_colour

                    # make the final label for this angle
                    legend = legend + " " + repr([int(x) for x in angle])
                    # add this to the list of terms to draw for this junction
                    to_draw_by_junction[junction_bond].append(
                        {
                            "legend": legend,
                            "Bonds": [bond1_idx, bond2_idx],
                            "Bond Colours": {
                                bond1_idx: bond_colour,
                                bond2_idx: bond_colour,
                            },
                            "Atoms": angle,
                        }
                    )

            for torsion in ff_labels["ProperTorsions"].keys():
                torsion_set = frozenset(torsion)
                if junction_atom in torsion_set and any(
                    dummy in torsion_set for dummy in junction_dummies
                ):
                    # this is a dihedral crossing the junction
                    # check if it has been removed, if removed just skip
                    if torsion_set in correction_data.removed_dihedrals:
                        continue

                    # if this dihedral has been kept we need to draw it
                    p1, p2, p3, p4 = torsion
                    # find the bonds for the dihedral to highlight
                    bond1_idx = rdkit_mol.GetBondBetweenAtoms(p1, p2).GetIdx()
                    bond2_idx = rdkit_mol.GetBondBetweenAtoms(p2, p3).GetIdx()
                    bond3_idx = rdkit_mol.GetBondBetweenAtoms(p3, p4).GetIdx()
                    # find the junction bond this should be stored under
                    junction_bond = [
                        atom
                        for atom in torsion
                        if atom == junction_atom or atom in junction_dummies
                    ]
                    # this should always be 2 atoms unless it is a torsion spanning two dummy groups
                    # in this case assign it to any
                    if len(junction_bond) != 2:
                        # make sure the extras are all dummies
                        temp_bond = [
                            atom for atom in junction_bond if atom in junction_dummies
                        ]
                        assert set(junction_bond) - set(temp_bond) == {
                            junction_atom
                        }, f"Issue determining junction bond for torsion:{torsion} with dummies {junction_dummies} and junction atom:{junction_atom}"
                        # select the junction atom and a dummy as the junction bond
                        # use a deterministic method
                        junction_bond = [sorted(temp_bond)[0], junction_atom]

                    junction_bond = frozenset(junction_bond)

                    # work out the colour and label for this dihedral
                    if torsion_set in correction_data.stiffened_dihedrals:
                        legend = "Stiffened Dihedral"
                        bond_colour = stiffened_term_colour
                    else:
                        legend = "Dihedral"
                        bond_colour = normal_term_colour

                    # make the final label for this dihedral
                    legend = legend + " " + repr([int(x) for x in torsion])
                    # add this to the list of terms to draw for this junction
                    to_draw_by_junction[junction_bond].append(
                        {
                            "legend": legend,
                            "Bonds": [bond1_idx, bond2_idx, bond3_idx],
                            "Bond Colours": {
                                bond_idx: bond_colour
                                for bond_idx in [bond1_idx, bond2_idx, bond3_idx]
                            },
                            "Atoms": torsion,
                        }
                    )

            for improper in ff_labels["ImproperTorsions"].keys():
                improper_set = frozenset(improper)
                if junction_atom in improper_set and any(
                    dummy in improper_set for dummy in junction_dummies
                ):
                    # this is an improper crossing the junction
                    # check if it has been removed, if removed just skip
                    if improper_set in correction_data.removed_impropers:
                        continue

                    # if this improper has been kept we need to draw it
                    p1, p2, p3, p4 = improper
                    # find the bonds for the improper to highlight assuming that the second index is the central atom
                    bond1_idx = rdkit_mol.GetBondBetweenAtoms(p2, p1).GetIdx()
                    bond2_idx = rdkit_mol.GetBondBetweenAtoms(p2, p3).GetIdx()
                    bond3_idx = rdkit_mol.GetBondBetweenAtoms(p2, p4).GetIdx()
                    # find the junction bond this should be stored under
                    junction_bond = [
                        atom
                        for atom in improper
                        if atom == junction_atom or atom in junction_dummies
                    ]

                    junction_bond = frozenset(junction_bond)

                    # make the final label for this angle
                    legend = "Improper " + repr([int(x) for x in improper])

                    # add this to the list of terms to draw for this junction
                    to_draw_by_junction[junction_bond].append(
                        {
                            "legend": legend,
                            "Bonds": [bond1_idx, bond2_idx, bond3_idx],
                            "Bond Colours": {
                                bond1_idx: normal_term_colour,
                                bond2_idx: normal_term_colour,
                                bond3_idx: normal_term_colour,
                            },
                            "Atoms": improper,
                        }
                    )

        # now we have all the terms to draw grouped by junction we can draw them
        # set up containers for drawing
        molecules = []
        highlight_atoms = []
        highlight_bonds = []
        bond_colours = []
        atom_colours = []
        legends = []

        # set up the atom colours
        atom_to_color = {
            atom: core_atom_colour for atom in core_atoms
        }  # set the junction atom colour
        for dummy in dummy_atoms:
            atom_to_color[dummy] = (
                dummy_atom_colour  # set the dummy atom colours on all dummies
            )

        # add a copy of the molecule with the atoms involved in the term annotated with their map numbers
        # deduplicate by unqiue interactions, as repeats can happen when terms connect two dummy groups
        seen_highlights = set()
        for highlight_junction in to_draw_by_junction.values():
            for highlight_data in highlight_junction:
                if frozenset(highlight_data["Atoms"]) in seen_highlights:
                    continue
                highlight_atoms.append(highlight_data["Atoms"])
                highlight_bonds.append(highlight_data["Bonds"])
                bond_colours.append(highlight_data["Bond Colours"])
                atom_colours.append(atom_to_color)
                legends.append(highlight_data["legend"])
                mol_copy = copy.deepcopy(rdkit_mol)
                atoms = highlight_data["Atoms"]
                for atom_idx in atoms:
                    atom = mol_copy.GetAtomWithIdx(atom_idx)
                    atom.SetProp("molAtomMapNumber", str(atom_idx))
                molecules.append(mol_copy)
                seen_highlights.add(frozenset(highlight_data["Atoms"]))

        res = Draw.MolsToGridImage(
            molecules,
            molsPerRow=4,
            highlightAtomLists=highlight_atoms,
            highlightBondLists=highlight_bonds,
            highlightBondColors=bond_colours,
            highlightAtomColors=atom_colours,
            subImgSize=(350, 350),
            legends=legends,
            useSVG=True,
        )
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / f"{ligand.name}_corrections.svg", "w") as fh:
            fh.write(res)


def _collect_junction_dihedrals(
    junction: JunctionData,
    force_field_labels: dict[str, Any],
    core_atoms: set[int],
) -> tuple[set[tuple], set[tuple]]:
    """
    Classify proper torsions for a junction into two categories.

    Parameters
    ----------
    junction : JunctionData
        The junction to consider when classifying dihedrals.
    force_field_labels : dict[str, Any]
        The force field labels used to identify dihedrals.
    core_atoms : set[int]
        The atoms which form the core region includes all core atoms in the molecule.

    Returns
    -------
    tuple[set, set]
        junction_dihedrals : set of (dummy, junction, phys_neighbor, other_core) tuples —
            crosses the junction bond, terminates deeper in the core.
        dummy_group_dihedrals : set of (deeper_dummy, d, junction, phys_neighbor) tuples —
            originates inside the dummy subtree, crosses the junction bond into the core.

    Notes
    -----
    Dihedrals whose central bond is {junction_atom, phys_neighbor} are intentionally
    excluded from both categories: those couple two dummy groups on opposite sides of the
    junction and are handled by _remove_cross_dummy_group_coupling.
    """
    junction_atom = junction.junction_atom
    dummy_atoms = junction.dummies
    physical_atoms = junction.physical
    other_core_atoms = core_atoms - physical_atoms - {junction_atom}

    junction_dihedrals: set[tuple] = set()
    dummy_group_dihedrals: set[tuple] = set()

    for dihedral in force_field_labels["ProperTorsions"].keys():
        has_junction = junction_atom in dihedral
        has_dummy = dummy_atoms.intersection(dihedral)
        has_physical = physical_atoms.intersection(dihedral)
        has_other_core = other_core_atoms.intersection(dihedral)

        if has_junction and has_dummy and has_physical and (has_other_core or len(has_physical) == 2):
            # (dummy, junction, phys_neighbor, other_core) — standard junction dihedral or
            # (dummy, junction, phys_neighbor, phys_neighbor) - torsion through a 3 membered ring
            junction_dihedrals.add(dihedral)
        elif has_junction and has_dummy and has_physical and not has_other_core:
            # Skip dihedrals whose central bond is {junction_atom, phys_neighbor}:
            # those link two dummy groups through the junction and are handled separately.
            central_atoms = {dihedral[1], dihedral[2]}
            if junction_atom in central_atoms and central_atoms.intersection(physical_atoms):
                continue
            dummy_group_dihedrals.add(dihedral)

    return junction_dihedrals, dummy_group_dihedrals


def _get_reachable_dummy_atoms(
    junction: JunctionData,
    dummy_atoms: set[int],
    rdkit_mol: Chem.Mol,
) -> set[int]:
    """
    Get all dummy atoms reachable from within this junction without crossing a core atom.

    Parameters
    ----------
    junction : JunctionData
        The junction to consider.
    dummy_atoms : set[int]
        All dummy atoms in the molecule.
    rdkit_mol : Chem.Mol
        The molecule which the junction belongs to.

    Returns
    -------
    set[int]
        All dummy atoms reachable from the junction without traversing core atoms.
    """
    visited: set[int] = set()
    dummy_queue = list(junction.dummies)
    while dummy_queue:
        atom_idx = dummy_queue.pop(0)
        if atom_idx in visited:
            continue
        visited.add(atom_idx)
        for neighbor in rdkit_mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nb_idx = neighbor.GetIdx()
            if nb_idx in dummy_atoms and nb_idx not in visited:
                dummy_queue.append(nb_idx)
    return visited


def _harmonize_junction_anchors(
    candidates_per_junction: dict[int, set[int]],
    rdkit_mol: Chem.Mol,
) -> dict[int, int]:
    """
    Greedily assign other_core terminal atoms to junctions to maximise shared anchors.

    At each step the terminal appearing in the most unassigned junctions' candidate
    sets is chosen and assigned to all those junctions simultaneously.  Ties are broken
    by atom mass (heavier preferred) then by index (higher preferred), to ensure a
    deterministic selection between repeats.

    Parameters
    ----------
    candidates_per_junction : dict[int, set[int]]
        Mapping from junction_atom index to the set of candidate other_core terminal
        atoms derived from that junction's junction_dihedrals.
    rdkit_mol : Chem.Mol

    Returns
    -------
    dict[int, int]
        Mapping from junction_atom index to the chosen terminal atom index.
        Junctions with empty candidate sets are omitted.
    """
    from collections import Counter
    assigned: dict[int, int] = {}
    unassigned = {j for j, cands in candidates_per_junction.items() if cands}

    while unassigned:
        # count the number of times each candidate is available
        freq: Counter[int] = Counter()
        for j in unassigned:
            for t in candidates_per_junction[j]:
                freq[t] += 1

        # get the best candidate
        max_freq = max(freq.values())
        best_terminal = max(
            (t for t, f in freq.items() if f == max_freq),
            key=lambda t: (rdkit_mol.GetAtomWithIdx(t).GetMass(), t),
        )

        # set the best candidate on all junctions it is valid for
        for j in list(unassigned):
            if best_terminal in candidates_per_junction[j]:
                assigned[j] = best_terminal
                unassigned.discard(j)

    return assigned


def _remove_cross_dummy_group_coupling(
    junctions: list[JunctionData],
    dummy_atoms: set[int],
    force_field_labels: dict[str, Any],
    rdkit_mol: Chem.Mol,
) -> CorrectionData:
    """
    Remove proper torsions that couple two distinct dummy groups through physical core atoms.

    Each junction owns a dummy group (all dummy atoms reachable without crossing into
    the core).  Any dihedral whose atom set spans two or more of these groups must pass
    through a physical core atom and is therefore removed.  Intra-group dihedrals (all
    dummy atoms belong to a single group) are left untouched. This removes torsions linking two dummy groups only passing
    through the core junction atom as well.

    Parameters
    ----------
    junctions : list[JunctionData]
        The dummy junctions in the molecule.
    dummy_atoms : set[int]
        All dummy atoms in the molecule.
    force_field_labels : dict[str, Any]
        The force field labels for the molecule, used to find the proper torsions which span dummy groups.
    rdkit_mol : Chem.Mol
        The molecule the junctions belong to.

    Returns
    -------
    CorrectionData
    """
    corrections = CorrectionData()

    if len(junctions) < 2:
        return corrections

    dummy_groups = [
        _get_reachable_dummy_atoms(j, dummy_atoms, rdkit_mol) for j in junctions
    ]

    # reverse map: dummy atom -> group index
    dummy_to_group: dict[int, int] = {}
    for group_id, group in enumerate(dummy_groups):
        for atom in group:
            dummy_to_group[atom] = group_id

    for dihedral in force_field_labels["ProperTorsions"].keys():
        groups_touched = {dummy_to_group[a] for a in dihedral if a in dummy_to_group}
        if len(groups_touched) > 1:
            corrections.removed_dihedrals.add(frozenset(dihedral))

    return corrections


def _filter_candidates_for_planar_ring_junction(
    junction_atom: int,
    candidates: set[int],
    rdkit_mol: Chem.Mol,
) -> set[int]:
    """
    Restrict dihedral terminal candidates to ring members for a planar junction in a ring.

    When a junction atom is SP2 and lies within a ring, the out-of-plane position
    of attached dummy atoms is best anchored by a torsion that terminates inside
    the same ring.  Choosing an in-ring terminal ensures the stiffened dihedral
    can be set with a single minimum at 180-degrees which is hardcoded in the HTF construction.

    Parameters
    ----------
    junction_atom : int
        Index of the junction atom (must be SP2 and in a ring before calling).
    candidates : set[int]
        Candidate other_core terminal atoms collected from the junction's
        junction_dihedrals.
    rdkit_mol : Chem.Mol
        The molecule this junction belongs to used for getting the hybridization and ring membership of the junction atom.

    Returns
    -------
    set[int]
        Candidates filtered to atoms that share a ring with junction_atom.
        Falls back to the original unfiltered candidates if none of the candidates
        are ring members, so that the junction is never left without an anchor.
    """
    junction_rings = [
        set(ring)
        for ring in rdkit_mol.GetRingInfo().AtomRings()
        if junction_atom in ring
    ]
    ring_candidates = {
        t for t in candidates
        if any(t in ring for ring in junction_rings)
    }
    return ring_candidates if ring_candidates else candidates


def _derive_uniform_valence_pruning(
    mapping: LigandAtomMapping, force_field: str
) -> dict[str, CorrectionData]:
    """
    Derive dummy valence term pruning applying uniform single-bond-angle-dihedral
    rules to every junction type, regardless of the number of physical neighbors.

    For each dummy junction the same algorithm is applied:

    - Keep one angle per dummy atom at the junction: (dummy, junction, phys_anchor)
    - Keep one junction dihedral per dummy atom: (dummy, junction, phys_anchor, other_core)
    - Keep one dummy-group dihedral per deeper dummy atom: (d2, d1, junction, phys_anchor)
    - The physical anchor is determined by the heaviest other_core terminal
    - Anchor selection is globally harmonised across all junctions so that nearby
      dummy groups prefer the same core terminal, minimising the number of core atoms used as anchors
    - Junction dihedrals at planar (SP2) junction atoms are stiffened to hold the
      dummy atom in the molecular plane; for SP2 junctions within a ring the anchor
      terminal is chosen from ring members where possible to provide a rigid
      reference frame
    - Dihedrals coupling two distinct dummy groups through physical core atoms are removed
    - Improper torsions spanning the core-dummy junction beyond the junction atom are removed
    - Connections within a single dummy group are never pruned

    Parameters
    ----------
    mapping : LigandAtomMapping
        The atom mapping between the two ligands, used to identify core and dummy atoms
        at each end state.
    force_field : str
        SMIRNOFF force field name (e.g. "openff-2.3.0.offxml") or XML string.

    Returns
    -------
    dict[str, CorrectionData]
        "lambda_0" corrections reference componentB atoms;
        "lambda_1" corrections reference componentA atoms.

    Notes
    -----
    - Currently only works with SMIRNOFF force fields.
    - Dummy-core junctions within the same ring system are skipped (ring openings are
      not supported).

    TODO
    ----
    - Check if a torsion ends in a collinear group or a group which becomes collinear.
    """
    print("deriving corrections using the uniform valence pruning scheme")
    all_corrections: dict[str, CorrectionData] = {
        "lambda_0": CorrectionData(),
        "lambda_1": CorrectionData(),
    }

    if not force_field.startswith("<?xml") and not force_field.endswith(".offxml"):
        force_field = force_field + ".offxml"
    ff = ForceField(force_field)

    for state_key in ["lambda_0", "lambda_1"]:
        if state_key == "lambda_0":
            smc = mapping.componentB
            core_atoms = set(mapping.componentA_to_componentB.values())
        else:
            smc = mapping.componentA
            core_atoms = set(mapping.componentA_to_componentB.keys())

        rdkit_mol = smc.to_openff().to_rdkit()
        dummy_atoms = {
            a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetIdx() not in core_atoms
        }
        ff_labels = ff.label_molecules(smc.to_openff().to_topology())[0]
        junctions = _find_dummy_junctions_rdkit(rdkit_mol, dummy_atoms)

        if not junctions:
            continue

        # classify proper torsions for every junction
        # as a junction dihedral originating in the junction dummy
        # or a dummy group dihedral from deeper within the dummy group and passing through the junction
        all_jd: dict[int, set[tuple]] = {}
        all_dgd: dict[int, set[tuple]] = {}
        for junction in junctions:
            jd, dgd = _collect_junction_dihedrals(junction, ff_labels, core_atoms)
            all_jd[junction.junction_atom] = jd
            all_dgd[junction.junction_atom] = dgd

        # collect candidate terminal atoms per junction, filtering to in-ring atoms
        # first for SP2 junctions that sit within a ring so that the stiffened anchor
        # dihedral terminates in a geometrically rigid ring member
        candidates_per_junction: dict[int, set[int]] = {}
        for junction in junctions:
            j_atom = junction.junction_atom
            excluded = junction.dummies | junction.physical | {j_atom}
            candidates: set[int] = set()
            for dihedral in all_jd[j_atom]:
                for a in dihedral:
                    if a not in excluded:
                        candidates.add(a)
            j_rdkit_atom = rdkit_mol.GetAtomWithIdx(j_atom)
            if (
                candidates
                and j_rdkit_atom.GetHybridization() == Chem.HybridizationType.SP2
                and j_rdkit_atom.IsInRing()
            ):
                candidates = _filter_candidates_for_planar_ring_junction(
                    j_atom, candidates, rdkit_mol
                )
            candidates_per_junction[j_atom] = candidates

        # globally harmonise anchor selection to maximise shared core atoms
        terminal_by_junction = _harmonize_junction_anchors(candidates_per_junction, rdkit_mol)

        # Resolve each chosen terminal to the physical neighbor that bridges junction→terminal
        phys_anchor_by_junction: dict[int, int | None] = {}
        for junction in junctions:
            j_atom = junction.junction_atom
            terminal = terminal_by_junction.get(j_atom)
            if terminal is None:
                phys_anchor_by_junction[j_atom] = None
                continue
            phys_candidates = [
                a.GetIdx()
                for a in rdkit_mol.GetAtomWithIdx(terminal).GetNeighbors()
                if a.GetIdx() in junction.physical
            ]
            phys_anchor_by_junction[j_atom] = phys_candidates[0] if phys_candidates else None

        # apply per-junction corrections
        for junction in junctions:
            j_atom = junction.junction_atom
            terminal = terminal_by_junction.get(j_atom)
            phys_to_keep = phys_anchor_by_junction[j_atom]
            corrections = CorrectionData()

            if terminal is not None and phys_to_keep is not None:
                # Remove junction dihedrals not ending in the chosen terminal
                for dihedral in all_jd[j_atom]:
                    if terminal not in dihedral:
                        corrections.removed_dihedrals.add(frozenset(dihedral))

                # Prune multiple-path redundancies (e.g. arising from 4-membered rings)
                corrections = _prune_multiple_path_dihedrals(all_jd[j_atom], corrections)

                # Stiffen the kept junction dihedral if the junction atom is planar (SP2) and in a ring:
                # the dummy must be held in the molecular plane and a normal torsion does
                # not provide a sufficient restoring force alone.
                j_atom_rdkit = rdkit_mol.GetAtomWithIdx(j_atom)
                if j_atom_rdkit.GetHybridization() == Chem.HybridizationType.SP2 and j_atom_rdkit.IsInRing():
                    for dihedral in all_jd[j_atom]:
                        if frozenset(dihedral) not in corrections.removed_dihedrals:
                            corrections.stiffened_dihedrals.add(frozenset(dihedral))

                # Remove angles not routed through the chosen physical anchor
                for angle in ff_labels["Angles"].keys():
                    if (
                        junction.dummies.intersection(angle)
                        and j_atom in angle
                        and junction.physical.intersection(angle)
                        and phys_to_keep not in angle
                    ):
                        corrections.removed_angles.add(frozenset(angle))

                # Remove dummy-group dihedrals not routed through the chosen physical anchor
                for dihedral in all_dgd[j_atom]:
                    if phys_to_keep not in dihedral:
                        corrections.removed_dihedrals.add(frozenset(dihedral))

            # Remove improper torsions coupling dummy and core beyond the junction atom
            corrections.merge(_find_improper_corrections(junction, ff_labels))

            if not _check_dummy_junction(
                junction=junction,
                corrections=corrections,
                force_field_labels=ff_labels,
                core_atoms=core_atoms,
            ):
                raise ValueError(
                    f"Uniform pruning failed validation for junction at atom {j_atom} "
                    f"with physical atoms {junction.physical} and dummy atoms {junction.dummies}. "
                    f"Corrections: {corrections}"
                )

            all_corrections[state_key].merge(corrections)

        # remove dihedrals coupling two distinct dummy groups through core atoms
        all_corrections[state_key].merge(
            _remove_cross_dummy_group_coupling(
                junctions=junctions,
                dummy_atoms=dummy_atoms,
                force_field_labels=ff_labels,
                rdkit_mol=rdkit_mol,
            )
        )

    return all_corrections