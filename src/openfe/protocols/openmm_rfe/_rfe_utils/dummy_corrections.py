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
                    print(
                        f"Deriving corrections for higher order junction with physical atoms {junction.physical} and dummy atoms {junction.dummies}"
                    )
                    corrections = _derive_higher_order_corrections(
                        junction, rdkit_mol, ff_labels, core_atoms
                    )
                    print(
                        f"Derived corrections for higher order junctions: {corrections}"
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

    # check if we should stiffen the dihedral based on the hybridisation of the junction atom
    # ring membership of the junction atom or if the terminal atom is in a free rotor
    junction_hybridisation = rdkit_mol.GetAtomWithIdx(junction_atom).GetHybridization()
    junction_ring = rdkit_mol.GetAtomWithIdx(junction_atom).IsInRing()
    if junction_hybridisation == Chem.HybridizationType.SP2 and junction_ring:
        stiffen_planar = True
    else:
        stiffen_planar = False

    # remove all dihedrals which do not terminate in the heaviest terminal atom
    for dihedral in dummy_junction_dihedrals:
        if heaviest_terminal_atom not in dihedral:
            corrections.removed_dihedrals.add(frozenset(dihedral))
        # we need to stiffen the dihedral if we want to keep the dummy group in plane with the rest of the molecule
        # or if we have to anchor using a free rotor
        elif heaviest_terminal_atom in dihedral and (heaviest_terminal_atom in rotor_atoms or stiffen_planar):
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

    # remove redundant angles not running through the kept dihedral to minimise the number of psychical anchor atoms
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

        # make sure we found 3 angles to soften per dummy atom at the junction
        if len(corrections.softened_angles) != 3 * len(dummy_atoms):
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
                # remove these as the geometry of the triple correction is distorted
                central_atoms = {dihedral[1], dihedral[2]}
                if junction_atom in central_atoms and central_atoms.intersection(
                        physical_atoms
                ):
                    corrections.removed_dihedrals.add(frozenset(dihedral))
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

def _derive_higher_order_corrections(
    junction: JunctionData,
    rdkit_mol: Chem.Mol,
    force_field_labels: dict[str, Any],
    core_atoms: set[int],
) -> CorrectionData:
    """Derive higher order corrections using the best practices from Fleck et al.

    Notes
    -----
    - We remove redundant physical atoms from the junction until we can pass to the triple junction function.
    - Atoms retained should be able to support a dihedral anchor if required
    """
    corrections = CorrectionData()
    junction_atom = junction.junction_atom
    physical_atoms = junction.physical
    dummy_atoms = junction.dummies

    # track the physical atoms which could support a dihedral anchor
    physicals_to_keep = set()
    other_core_atoms = core_atoms - physical_atoms - {junction_atom}
    print(force_field_labels)
    for dihedral in force_field_labels["ProperTorsions"].keys():
        # do checks to classify the dihedral
        has_junction = junction_atom in dihedral
        has_dummy = dummy_atoms.intersection(dihedral)
        has_physical = physical_atoms.intersection(dihedral)
        has_other_core = other_core_atoms.intersection(dihedral)

        if has_junction and has_dummy and has_physical and has_other_core:
            physicals_to_keep.update(physical_atoms.intersection(dihedral))

    # keep the first 3 physical atoms that can support an anchor
    new_physical_atoms = physicals_to_keep.intersection(physical_atoms)
    print(f"Found physical atoms which support a dihedral anchor {new_physical_atoms}")
    if len(new_physical_atoms) > 3:
        print("Filtering higher order physical atoms")
        # deterministically remove extra atoms
        new_physical_atoms = set(sorted(new_physical_atoms)[:3])
    elif len(new_physical_atoms) < 3:
        print(f"Padding higher order physical atoms")
        # add the junction physical atoms to the list till we hit length 3 in a deterministic way
        new_physical_atoms = new_physical_atoms.union(
            set(sorted(physical_atoms)[: 3 - len(new_physical_atoms)])
        )

    # the removed physical atoms
    removed_physical_atoms = physical_atoms - new_physical_atoms
    print(f"Redundant physical atoms {removed_physical_atoms}")
    # we now need to remove all angles which couple the removed physical atoms to the dummies at this junction
    for angle in force_field_labels["Angles"].keys():
        # classify the angle
        has_junction_a = junction_atom in angle
        has_dummy_a = dummy_atoms.intersection(angle)
        has_removed_phys = removed_physical_atoms.intersection(angle)

        if has_junction_a and has_dummy_a and has_removed_phys:
            corrections.removed_angles.add(frozenset(angle))
    print(f"Redundant angles removed {corrections.removed_angles}")

    # now we need to remove all dihedrals which involve a dummy atom and pass through the removed physical atoms
    # this includes dummy junction and dummy group anchors
    for dihedral in force_field_labels["ProperTorsions"].keys():
        has_junction = junction_atom in dihedral
        has_dummy = dummy_atoms.intersection(dihedral)
        has_removed_phys = removed_physical_atoms.intersection(dihedral)

        if has_junction and has_dummy and has_removed_phys:
            corrections.removed_dihedrals.add(frozenset(dihedral))

    print(f"Redundant dihedrals removed {corrections.removed_dihedrals}")

    # make the new junction and pass it to the triple connection function
    new_junction = JunctionData(
        junction_atom=junction_atom,
        dummies=dummy_atoms,
        physical=new_physical_atoms,
    )
    print(f"Higher order junction reduced to {new_junction}")
    triple_corrections = _derive_triple_corrections(
        new_junction, rdkit_mol, force_field_labels, core_atoms
    )
    # merge the corrections
    corrections.merge(triple_corrections)
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
