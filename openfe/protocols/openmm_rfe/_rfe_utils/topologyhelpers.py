# This code is in parts based on TopologyProposal in perses
# (https://github.com/choderalab/perses)
# The eventual goal is to move this to the OpenFE alchemical topology
# building toolsets.
# LICENSE: MIT

from copy import deepcopy
import itertools
import logging
import warnings

import mdtraj as mdt
from mdtraj.core.residue_names import _SOLVENT_TYPES
import numpy as np
from openmm import app
from openmm import unit as omm_unit
from openff.unit import unit


logger = logging.getLogger(__name__)


def get_alchemical_water(topology, positions,
                         charge_correction,
                         distance_cutoff=0.8 * unit.nanometer):
    """
    Based off perses.ultils.charge_changing.get_water_indices.

    Parameters
    ----------
    topology : ??
    positions : ??
    charge_correction: bool
    """
    # Exit early and return Nones if you don't actually want to do this
    if not charge_correction:
        return None, None

    # construct a new 
    traj = mdt.Trajectory(
        positions[np.newaxis, ...],
        mdt.Topology.from_openmm(topology)
    )

    water_atoms = traj.topology.select("water")
    solvent_residue_names = list(_SOLVENT_TYPES)
    solute_atoms = [atom.index for atom in traj.topology.atom
                    if atom.residue.name not in solvent_residue_names]

    excluded_waters = mdt.compute_neighbors(
        traj, distance_cutoff.to(unit.nanometer).m,
        solute_atoms, haystack_indices=water_atoms
    )[0]

    solvent_indices = set([
        atom.residue.index for atom in traj.topology.atoms
        if (atom.index in water_atoms) and (atom.index not in excluded_waters)
    ])

    if len(sovlent_indices) < 0:
        errmsg = ("There are no waters outside of a "
                  f"{distance_cutoff.to(unit.nanometer) nanometer distance "
                  "of the system solutes to be used as alchemical waters")
        raise ValueError(errmsg)

    # choose the water further away
    
    


def combined_topology(topology1, topology2, exclude_resids=None):
    """
    Create a new topology combining these two topologies.

    The box information from the *first* topology will be copied over

    Parameters
    ----------
    topology1 : openmm.app.Topology
    topology2 : openmm.app.Topology
    exclude_resids : npt.NDArray
      Residue indices in topology 1 to exclude from the combined topology.

    Returns
    -------
    new : openmm.app.Topology
    appended_resids : npt.NDArray
      Residue indices of the residues appended from topology2 in the new
      topology.
    """
    if exclude_resids is None:
        exclude_resids = []

    top = app.Topology()

    # create list of excluded residues from topology
    excluded_res = [
        r for r in topology1.residues() if r.index in exclude_resids
    ]

    # get a list of all excluded atoms
    excluded_atoms = set(itertools.chain.from_iterable(
        r.atoms() for r in excluded_res)
    )

    # add new copies of selected chains, residues, and atoms; keep mapping
    # of old atoms to new for adding bonds later
    old_to_new_atom_map = {}
    appended_resids = []
    for chain_id, chain in enumerate(
            itertools.chain(topology1.chains(), topology2.chains())):
        # TODO: is chain ID int or str? I recall it being int in MDTraj....
        # are there any issues if we just add a blank chain?
        new_chain = top.addChain(chain_id)
        for residue in chain.residues():
            if residue in excluded_res:
                continue

            new_res = top.addResidue(residue.name,
                                     new_chain,
                                     residue.id)

            # append the new resindex if it's part of topology2
            if residue in list(topology2.residues()):
                appended_resids.append(new_res.index)

            for atom in residue.atoms():
                new_atom = top.addAtom(atom.name,
                                       atom.element,
                                       new_res,
                                       atom.id)
                old_to_new_atom_map[atom] = new_atom

    # figure out which bonds to keep: drop any that involve removed atoms
    def atoms_for_bond(bond):
        return {bond.atom1, bond.atom2}

    keep_bonds = (bond for bond in itertools.chain(topology1.bonds(),
                                                   topology2.bonds())
                  if not (atoms_for_bond(bond) & excluded_atoms))

    # add bonds to topology
    for bond in keep_bonds:
        top.addBond(old_to_new_atom_map[bond.atom1],
                    old_to_new_atom_map[bond.atom2],
                    bond.type,
                    bond.order)

    # Copy over the box vectors
    top.setPeriodicBoxVectors(topology1.getPeriodicBoxVectors())

    return top, np.array(appended_resids)


def _get_indices(topology, resids):
    """
    Get the atoms indices from an array of residue indices in an OpenMM Topology

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to search from.
    residue_name : str
        Name of the residue to get the indices for.
    """
    # TODO: remove, this shouldn't be necessary anymore
    if len(resids) > 1:
        raise ValueError("multiple residues were found")

    # create list of openmm residues
    top_res = [r for r in topology.residues() if r.index in resids]

    # get a list of all atoms in residues
    top_atoms = list(itertools.chain.from_iterable(r.atoms() for r in top_res))

    return [at.index for at in top_atoms]


def _remove_constraints(old_to_new_atom_map, old_system, old_topology,
                        new_system, new_topology):
    """
    Adapted from Perses' Topology Proposal. Adjusts atom mapping to account for
    any bonds that are constrained but change in length.

    Parameters
    ----------
    old_to_new_atom_map : dict of int : int
        Atom mapping between the old and new systems.
    old_system : openmm.app.System
        System of the "old" alchemical state.
    old_topology : openmm.app.Topology
        Topology of the "old" alchemical state.
    new_system : openmm.app.System
        System of the "new" alchemical state.
    new_topology : openmm.app.Topology
        Topology of the "new" alchemical state.

    Returns
    -------
    no_const_old_to_new_atom_map : dict of int : int
        Adjusted version of the input mapping but with atoms involving changes
        in lengths of constrained bonds removed.

    TODO
    ----
    * Very slow, needs refactoring
    * Can we drop having topologies as inputs here?
    """
    no_const_old_to_new_atom_map = deepcopy(old_to_new_atom_map)

    h_elem = app.Element.getByAtomicNumber(1)
    old_H_atoms = {i for i, atom in enumerate(old_topology.atoms())
                   if atom.element == h_elem and i in old_to_new_atom_map}
    new_H_atoms = {i for i, atom in enumerate(new_topology.atoms())
                   if atom.element == h_elem and i in old_to_new_atom_map.values()}

    def pick_H(i, j, x, y) -> int:
        """Identify which atom to remove to resolve constraint violation

        i maps to x, j maps to y

        Returns either i or j (whichever is H) to remove from mapping
        """
        if i in old_H_atoms or x in new_H_atoms:
            return i
        elif j in old_H_atoms or y in new_H_atoms:
            return j
        else:
            raise ValueError(f"Couldn't resolve constraint demapping for atoms"
                             f" A: {i}-{j} B: {x}-{y}")

    old_constraints: dict[[int, int], float] = dict()
    for idx in range(old_system.getNumConstraints()):
        atom1, atom2, length = old_system.getConstraintParameters(idx)

        if atom1 in old_to_new_atom_map and atom2 in old_to_new_atom_map:
            old_constraints[atom1, atom2] = length

    new_constraints = dict()
    for idx in range(new_system.getNumConstraints()):
        atom1, atom2, length = new_system.getConstraintParameters(idx)

        if (atom1 in old_to_new_atom_map.values() and
                atom2 in old_to_new_atom_map.values()):
            new_constraints[atom1, atom2] = length

    # there are two reasons constraints would invalidate a mapping entry
    # 1) length of constraint changed (but both constrained)
    # 2) constraint removed to harmonic bond (only one constrained)
    to_del = []
    for (i, j), l_old in old_constraints.items():
        x, y = old_to_new_atom_map[i], old_to_new_atom_map[j]

        try:
            l_new = new_constraints.pop((x, y))
        except KeyError:
            try:
                l_new = new_constraints.pop((y, x))
            except KeyError:
                # type 2) constraint doesn't exist in new system
                to_del.append(pick_H(i, j, x, y))
                continue

        # type 1) constraint length changed
        if l_old != l_new:
            to_del.append(pick_H(i, j, x, y))

    # iterate over new_constraints (we were .popping items out)
    # (if any left these are type 2))
    if new_constraints:
        new_to_old = {v: k for k, v in old_to_new_atom_map.items()}

        for x, y in new_constraints:
            i, j = new_to_old[x], new_to_old[y]

            to_del.append(pick_H(i, j, x, y))

    for idx in to_del:
        del no_const_old_to_new_atom_map[idx]

    return no_const_old_to_new_atom_map


def get_system_mappings(old_to_new_atom_map,
                        old_system, old_topology, old_resids,
                        new_system, new_topology, new_resids,
                        fix_constraints=True):
    """
    From a starting alchemical map between two molecules, get the mappings
    between two alchemical end state systems.

    Optionally, also fixes the mapping to account for a) element changes, and
    b) changes in bond lengths for constraints.

    Parameters
    ----------
    old_to_new_atom_map : dict of int : int
        Atom mapping between the old and new systems.
    old_system : openmm.app.System
        System of the "old" alchemical state.
    old_topology : openmm.app.Topology
        Topology of the "old" alchemical state.
    old_resids : npt.NDArray
        Residue ids of the alchemical residues in the "old" topology.
    new_system : openmm.app.System
        System of the "new" alchemical state.
    new_topology : openmm.app.Topology
        Topology of the "new" alchemical state.
    new_resids : npt.NDArray
        Residue ids of the alchemical residues in the "new" topology.
    fix_constraints : bool, default True
        Whether to fix the atom mapping by removing any atoms which are
        involved in constrained bonds that change length across the alchemical
        change.

    Returns
    -------
    mappings : dictionary
        A dictionary with all the necessary mappings for the two systems.
        These include:
            1. old_to_new_atom_map
              This includes all the atoms mapped between the two systems
              (including non-core atoms, i.e. environment).
            2. new_to_old_atom_map
              The inverted dictionary of old_to_new_atom_map
            3. old_to_new_core_atom_map
              The atom mapping of the "core" atoms (i.e. atoms in alchemical
              residues) between the old and new systems
            4. new_to_old_core_atom_map
              The inverted dictionary of old_to_new_core_atom_map
            5. old_to_new_env_atom_map
              The atom mapping of solely the "environment" atoms between the
              old and new systems.
            6. new_to_old_env_atom_map
              The inverted dictionaryu of old_to_new_env_atom_map.
            7. old_mol_indices
              Indices of the alchemical molecule in the old system.
            8. new_mol_indices
              Indices of the alchemical molecule in the new system.
    """
    # Get the indices of the atoms in the alchemical residue of interest for
    # both the old and new systems
    old_at_indices = _get_indices(old_topology, old_resids)
    new_at_indices = _get_indices(new_topology, new_resids)

    # We assume that the atom indices are linear in the residue so we shift
    # by the index of the first atom in each residue
    adjusted_old_to_new_map = {}
    for (key, value) in old_to_new_atom_map.items():
        shift_old = old_at_indices[0] + key
        shift_new = new_at_indices[0] + value
        adjusted_old_to_new_map[shift_old] = shift_new

    # TODO: the original intent here was to apply over the full mapping of all
    # the atoms in the two systems. For now we are only doing the alchemical
    # residues. We might want to change this as necessary in the future.
    if not fix_constraints:
        wmsg = ("Not attempting to fix atom mapping to account for "
                "constraints. Please note that core atoms which have "
                "constrained bonds and changing bond lengths are not allowed.")
        warnings.warn(wmsg)
    else:
        adjusted_old_to_new_map = _remove_constraints(
            adjusted_old_to_new_map, old_system, old_topology,
            new_system, new_topology)

    # We return a dictionary with all the necessary mappings (as they are
    # needed downstream). These include:
    #  1. old_to_new_atom_map
    #     This includes all the atoms mapped between the two systems
    #     (including non-core atoms, i.e. environment).
    #  2. new_to_old_atom_map
    #     The inverted dictionary of old_to_new_atom_map
    #  3. old_to_new_core_atom_map
    #     The atom mapping of the "core" atoms (i.e. atoms in alchemical
    #     residues) between the old and new systems
    #  4. new_to_old_core_atom_map
    #     The inverted dictionary of old_to_new_core_atom_map
    #  5. old_to_new_env_atom_map
    #     The atom mapping of solely the "environment" atoms between the old
    #     and new systems.
    #  6. new_to_old_env_atom_map
    #     The inverted dictionaryu of old_to_new_env_atom_map.

    # Because of how we append the topologies, we can assume that the last
    # residue in the "new" topology is the ligand, just to be sure we check
    # this here - temp fix for now
    for at in new_topology.atoms():
        if at.index > new_at_indices[-1]:
            raise ValueError("residues are appended after the new ligand")

    # We assume that all the atoms up until the first ligand atom match
    # except from the indices of the ligand in the old topology.
    new_to_old_all_map = {}
    old_mol_offset = len(old_at_indices)
    for i in range(new_at_indices[0]):
        if i >= old_at_indices[0]:
            old_idx = i + old_mol_offset
        else:
            old_idx = i
        new_to_old_all_map[i] = old_idx

    # At this point we only have environment atoms so make a copy
    new_to_old_env_map = deepcopy(new_to_old_all_map)

    # Next we append the contents of the "core" map we already have
    for key, value in adjusted_old_to_new_map.items():
        # reverse order because we are going new->old instead of old->new
        new_to_old_all_map[value] = key

    # Now let's create our output dictionary
    mappings = {}
    mappings['new_to_old_atom_map'] = new_to_old_all_map
    mappings['old_to_new_atom_map'] = {v: k for k, v in new_to_old_all_map.items()}
    mappings['new_to_old_core_atom_map'] = {v: k for k, v in adjusted_old_to_new_map.items()}
    mappings['old_to_new_core_atom_map'] = adjusted_old_to_new_map
    mappings['new_to_old_env_atom_map'] = new_to_old_env_map
    mappings['old_to_new_env_atom_map'] = {v: k for k, v in new_to_old_env_map.items()}
    mappings['old_mol_indices'] = old_at_indices
    mappings['new_mol_indices'] = new_at_indices

    return mappings


def set_and_check_new_positions(mapping, old_topology, new_topology,
                                old_positions, insert_positions,
                                tolerance=1.0):
    """
    Utility to create new positions given a mapping, the old positions and
    the positions of the molecule being inserted, defined by `insert_positions.

    This will also softly check that the RMS distance between the core atoms
    of the old and new atoms do not differ by more than the amount specified
    by `tolerance`.

    Parameters
    ----------
    mapping : dict of int : int
        Dictionary of atom mappings between the old and new systems.
    old_topology : openmm.app.Topology
        Topology of the "old" alchemical state.
    new_topology : openmm.app.Topology
        Topology of the "new" alchemical state.
    old_positions : simtk.unit.Quantity
        Position of the "old" alchemical state.
    insert_positions : simtk.unit.Quantity
        Positions of the alchemically changing molecule in the "new" alchemical
        state.
    tolerance : float
        Warning threshold for deviations along any dimension (x,y,z) in mapped
        atoms between the "old" and "new" positions. Default 1.0.
    """
    # Get the positions in Angstrom as raw numpy arrays
    old_pos_array = old_positions.value_in_unit(omm_unit.angstrom)
    add_pos_array = insert_positions.value_in_unit(omm_unit.angstrom)

    # Create empty ndarray of size atoms to hold the positions
    new_pos_array = np.zeros((new_topology.getNumAtoms(), 3))

    # get your mappings
    new_idxs = list(mapping['old_to_new_atom_map'].values())
    old_idxs = list(mapping['old_to_new_atom_map'].keys())
    new_mol_idxs = mapping['new_mol_indices']

    # copy over the old positions for mapped atoms
    new_pos_array[new_idxs, :] = old_pos_array[old_idxs, :]
    # copy over the new alchemical molecule positions
    new_pos_array[new_mol_idxs, :] = add_pos_array

    # loop through all mapped atoms and make sure we don't deviate by more than
    # tolerance - not super necessary, but it's a nice sanity check
    for key, val in mapping['old_to_new_atom_map'].items():
        if np.any(
            np.abs(new_pos_array[val] - old_pos_array[key]) > tolerance):
            wmsg = f"mapping {key} : {val} deviates by more than {tolerance}"
            warnings.warn(wmsg)
            logging.warning(wmsg)

    return new_pos_array * omm_unit.angstrom
