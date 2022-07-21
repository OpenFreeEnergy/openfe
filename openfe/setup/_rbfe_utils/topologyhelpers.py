# This code is in parts based on TopologyProposal in perses
# (https://github.com/choderalab/perses)
# The eventual goal is to move this to the OpenFE alchemical topology
# building toolsets.
# LICENSE: MIT

import warnings
from copy import deepcopy
import numpy as np
from openmm import app, unit


def _append_topology(destination_topology, source_topology,
                     exclude_residue_name=None):
    """
    Originally from Perses.

    Add the source OpenMM Topology to the destination Topology.

    Parameters
    ----------
    destination_topology : openmm.app.Topology
        The Topology to which the contents of `source_topology` are to be
        added.
    source_topology : openmm.app.Topology
        The Topology to be added.
    exclude_residue_name : str, optional
        Any residues matching this name are excluded.

    Notes
    -----
    * Does not copy over periodic box vectors
    """
    if exclude_residue_name is None:
        # something with 3 characters that is never a residue name
        exclude_residue_name = "   "

    new_atoms = {}
    for chain in source_topology.chains():
        new_chain = destination_topology.addChain(chain.id)
        for residue in chain.residues():
            # TODO: should we use complete residue names?
            if (residue.name[:3] == exclude_residue_name[:3]):
                continue
            new_residue = destination_topology.addResidue(residue.name,
                                                          new_chain,
                                                          residue.id)
            for atom in residue.atoms():
                new_atom = destination_topology.addAtom(atom.name,
                                                        atom.element,
                                                        new_residue, atom.id)
                new_atoms[atom] = new_atom
    for bond in source_topology.bonds():
        if ((bond[0].residue.name[:3] == exclude_residue_name[:3]) or
                (bond[1].residue.name[:3] == exclude_residue_name[:3])):
            continue
        order = bond.order
        destination_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]],
                                     order=order)


def append_new_topology_item(old_topology, new_topology,
                             exclude_residue_name=None):
    """
    Create a "new" topology by appending the contents of ``new_topology`` to
    the ``old_topology``.

    Optionally exclude a given from ``old_topology`` on building.

    Parameters
    ----------
    old_topology : openmm.app.Topology
        Old topology to which the ``new_topology`` contents will be appended
        to.
    new_topology : openmm.app.Topology
        New topology item to be appended to ``old_topology``.
    exclude_residue_name : str, optional
        Name of residue in ``old_topology`` to be excluded in building the
        appended topology.

    Returns
    -------
    appended_topology : openmm.app.Topology
        Topology containing the combined old and new topology items, excluding
        any residues which were defined in ``exclude_residue_name``.
    """
    # Create an empty topology
    appended_topology = app.Topology()

    # Append old topology to new topology, excluding residue as required
    _append_topology(appended_topology, old_topology, exclude_residue_name)

    # Now we append the contents of the new topology
    _append_topology(appended_topology, new_topology)

    # Copy over the box vectors
    appended_topology.setPeriodicBoxVectors(
        old_topology.getPeriodicBoxVectors())

    return appended_topology


def _get_indices(topology, residue_name):
    """
    Get the indices of a unique residue in an OpenMM Topology

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to search from.
    residue_name : str
        Name of the residue to get the indices for.
    """
    residues = []
    for res in topology.residues():
        if res.name == residue_name:
            residues.append(res)

    if len(residues) > 1:
        raise ValueError("multiple residues were found")

    return [at.index for at in residues[0].atoms()]


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
    old_hydrogens = {
        at.index for at in old_topology.atoms() if at.element == h_elem}
    new_hydrogens = {
        at.index for at in new_topology.atoms() if at.element == h_elem}

    old_constraints = {}
    for idx in range(old_system.getNumConstraints()):
        atom1, atom2, length = old_system.getConstraintParameters(idx)
        if atom1 in old_hydrogens:
            old_constraints[atom1] = length
        if atom2 in old_hydrogens:
            old_constraints[atom2] = length

    new_constraints = {}
    for idx in range(new_system.getNumConstraints()):
        atom1, atom2, length = new_system.getConstraintParameters(idx)
        if atom1 in new_hydrogens:
            new_constraints[atom1] = length
        if atom2 in new_hydrogens:
            new_constraints[atom2] = length

    to_del = []
    for new_index, old_index in old_to_new_atom_map.items():
        # Constraints exist for both end atom states
        # TODO: check if we should be not allowing a fully dropped constraint
        if (new_index in new_constraints.keys() and
                old_index in old_constraints.keys()):
            # Check that the length of the constraints hasn't changed
            if not old_constraints[old_index] == new_constraints[new_index]:
                to_del.append(old_index)

    for idx in to_del:
        del no_const_old_to_new_atom_map[idx]

    return no_const_old_to_new_atom_map


def get_system_mappings(old_to_new_atom_map,
                        old_system, old_topology, old_residue,
                        new_system, new_topology, new_residue,
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
    old_residue : str
        Name of the alchemical residue in the "old" topology.
    new_system : openmm.app.System
        System of the "new" alchemical state.
    new_topology : openmm.app.Topology
        Topology of the "new" alchemical state.
    new_residue : str
        Name of the alchemical residue in the "new" topology.
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
    old_at_indices = _get_indices(old_topology, old_residue)
    new_at_indices = _get_indices(new_topology, new_residue)

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
                                shift_insert=None, tolerance=0.5):
    """
    Utility to create new positions given a mapping, the old positions and
    the positions of the molecule being inserted, defined by `insert_positions.

    This will also check that the RMS distance between the core atoms of the
    old and new atoms do not differ by more than the amount specified by
    `rms_tolerance`.

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
    shift_insert : np.ndarray (3,)
        Amount to shift `insert_positions` by in the x,y,z direction in
        Angstrom. Default None.
    tolerance : float
        Maximum allowed deviation along any dimension (x,y,z) in mapped atoms
        between the "old" and "new" positions. Default 0.5.
    """
    # Get the positions in Angstrom as raw numpy arrays
    old_pos_array = old_positions.value_in_unit(unit.angstrom)
    add_pos_array = insert_positions.value_in_unit(unit.angstrom)

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
    if shift_insert is not None:
        new_pos_array[new_mol_idxs, :] += shift_insert

    # loop through all mapped atoms and make sure we don't deviate by more than
    # tolerance - not super necessary, but it's a nice sanity check that should
    # eventually make it to a test
    for key, val in mapping['old_to_new_atom_map'].items():
        if np.any((new_pos_array[val] - old_pos_array[key]) > tolerance):
            errmsg = f"mapping {key} : {val} deviates by more than {tolerance}"
            raise ValueError(errmsg)

    return new_pos_array * unit.angstrom
