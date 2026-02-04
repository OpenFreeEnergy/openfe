# This code is in parts based on TopologyProposal in perses
# (https://github.com/choderalab/perses)
# The eventual goal is to move this to the OpenFE alchemical topology
# building toolsets.
# LICENSE: MIT

# turn off formatting since this is mostly vendored code
# fmt: off

import ast
from collections import defaultdict
import itertools
import logging
import warnings
from copy import deepcopy
import json
import math
from typing import Optional, Union

import mdtraj as mdt
import numpy as np
import numpy.typing as npt
from mdtraj.core.residue_names import _SOLVENT_TYPES
from openff.units import Quantity, unit
from openmm import NonbondedForce, System, app
from openmm import unit as omm_unit
import openmm

from openfe import SolventComponent
from openfe.protocols.openmm_rfe._rfe_utils.relative import HybridTopologyFactory

logger = logging.getLogger(__name__)


def _get_ion_and_water_parameters(
    topology: app.Topology,
    system: System,
    ion_resname: str,
    water_resname: str = 'HOH',
):
    """
    Get ion, and water (oxygen and hydrogen) atoms parameters.

    Parameters
    ----------
    topology : app.Topology
      The topology to search for the ion and water
    system : app.System
      The system associated with the input topology object.
    ion_resname : str
      The residue name of the ion to get parameters for
    water_resname : str
      The residue name of the water to get parameters for. Default 'HOH'.

    Returns
    -------
    ion_charge : float
      The partial charge of the ion atom
    ion_sigma : float
      The NonbondedForce sigma parameter of the ion atom
    ion_epsilon : float
      The NonbondedForce epsilon parameter of the ion atom
    o_charge : float
      The partial charge of the water oxygen.
    h_charge : float
      The partial charge of the water hydrogen.

    Raises
    ------
    ValueError
      If there are no ``ion_resname`` or ``water_resname`` named residues in
      the input ``topology``.

    Attribution
    -----------
    Based on `perses.utils.charge_changing.get_ion_and_water_parameters`.
    """
    def _find_atom(topology, resname, elementname):
        for atom in topology.atoms():
            if atom.residue.name == resname:
                if (elementname is None or atom.element.symbol == elementname):
                    return atom.index
        errmsg = ("Error encountered when attempting to explicitly handle "
                  "charge changes using an alchemical water. No residue "
                  f"named: {resname} found, with element {elementname}")
        raise ValueError(errmsg)

    ion_index = _find_atom(topology, ion_resname, None)
    oxygen_index = _find_atom(topology, water_resname, 'O')
    hydrogen_index = _find_atom(topology, water_resname, 'H')

    nbf = [i for i in system.getForces()
           if isinstance(i, NonbondedForce)][0]

    ion_charge, ion_sigma, ion_epsilon = nbf.getParticleParameters(ion_index)
    o_charge, _, _ = nbf.getParticleParameters(oxygen_index)
    h_charge, _, _ = nbf.getParticleParameters(hydrogen_index)

    return ion_charge, ion_sigma, ion_epsilon, o_charge, h_charge


def _fix_alchemical_water_atom_mapping(
    system_mapping: dict[str, Union[dict[int, int], list[int]]],
    b_idx: int,
) -> None:
    """
    In-place fix atom mapping to account for alchemical water.

    Parameters
    ----------
    system_mapping : dict
      Dictionary of system mappings.
    b_idx : int
      The index of the state B particle.
    """
    a_idx = system_mapping['new_to_old_atom_map'][b_idx]

    # Note, because these are already shared positions, we don't
    # append alchemical molecule indices in the new & old molecule
    # i.e. the `old_mol_indices` and `new_mol_indices` lists

    # remove atom from the environment atom map
    system_mapping['old_to_new_env_atom_map'].pop(a_idx)
    system_mapping['new_to_old_env_atom_map'].pop(b_idx)

    # add atom to the new_to_old_core atom maps
    system_mapping['old_to_new_core_atom_map'][a_idx] = b_idx
    system_mapping['new_to_old_core_atom_map'][b_idx] = a_idx


def handle_alchemical_waters(
    water_resids: list[int], topology: app.Topology,
    system: System, system_mapping: dict,
    charge_difference: int,
    solvent_component: SolventComponent,
):
    """
    Add alchemical waters from a pre-defined list.

    Parameters
    ----------
    water_resids : list[int]
      A list of alchemical water residues.
    topology : app.Topology
      The topology to search for the ion and water
    system : app.System
      The system associated with the input topology object.
    system_mapping : dictionary
      A dictionary of system mappings between the stateA and stateB systems
    charge_difference : int
      The charge difference between state A and state B.
    positive_ion_resname : str
      The name of a positive ion to replace the water with if the absolute
      charge difference is positive.
    negative_ion_resname : str
      The name of a negative ion to replace the water with if the absolute
      charge difference is negative.
    water_resname : str
      The residue name of the water to get parameters for. Default 'HOH'.

    Raises
    ------
    ValueError
      If the absolute charge difference is not equalent to the number of
      alchemical water resids.
      If the chosen alchemical water has virtual sites (i.e. is not
      a 3 site water molecule).

    Attribution
    -----------
    Based on `perses.utils.charge_changing.transform_waters_into_ions`.
    """

    if abs(charge_difference) != len(water_resids):
        errmsg = ("There should be as many alchemical water residues: "
                  f"{len(water_resids)} as the absolute charge "
                  f"difference: {abs(charge_difference)}")
        raise ValueError(errmsg)

    if charge_difference > 0:
        ion_resname = solvent_component.positive_ion.strip('-+').upper()
    elif charge_difference < 0:
        ion_resname = solvent_component.negative_ion.strip('-+').upper()
    # if there's no charge difference then just skip altogether
    else:
        return None

    ion_charge, ion_sigma, ion_epsilon, o_charge, h_charge = _get_ion_and_water_parameters(
        topology, system, ion_resname,
        'HOH',  # Modeller always adds HOH waters
    )

    # get the nonbonded forces
    nbfrcs = [i for i in system.getForces()
              if isinstance(i, NonbondedForce)]
    if len(nbfrcs) > 1:
        raise ValueError("Too many NonbondedForce forces found")

    # for convenience just grab the first & only entry
    nbf = nbfrcs[0]

    # Loop through residues, check if they match the residue index
    # mutate the atom as necessary
    for res in topology.residues():
        if res.index in water_resids:
            # if the number of atoms > 3, then we have virtual sites which are
            # not supported currently
            if len([at for at in res.atoms()]) > 3:
                errmsg = ("Non 3-site waters (i.e. waters with virtual sites) "
                          "are not currently supported as alchemical waters")
                raise ValueError(errmsg)

            for at in res.atoms():
                idx = at.index
                charge, sigma, epsilon = nbf.getParticleParameters(idx)
                _fix_alchemical_water_atom_mapping(system_mapping, idx)

                if charge == o_charge:
                    nbf.setParticleParameters(
                        idx, ion_charge, ion_sigma, ion_epsilon
                    )
                else:
                    if charge != h_charge:
                        errmsg = ("modifying an atom that doesn't match known "
                                  "water parameters")
                        raise ValueError(errmsg)

                    nbf.setParticleParameters(idx, 0.0, sigma, epsilon)


def get_alchemical_waters(
    topology: app.Topology,
    positions: npt.NDArray,
    charge_difference: int,
    distance_cutoff: Quantity = 0.8 * unit.nanometer,
) -> list[int]:
    """
    Pick a list of waters to be used for alchemical charge correction.

    Parameters
    ----------
    topology : openmm.app.Topology
      The topology to search for an alchemical water.
    positions : npt.NDArray
      The coordinates of the atoms associated with the ``topology``.
    charge_difference : int
      The charge difference between the two end states
      calculated as stateA_formal_charge - stateB_formal_charge.
    distance_cutoff : openff.units.Quantity
      The minimum distance away from the solutes from which an alchemical
      water can be chosen.


    Returns
    -------
    chosen_residues : list[int]
        A list of residue indices for each chosen alchemical water.

    Notes
    -----
    Based off perses.utils.charge_changing.get_water_indices.
    """
    # if the charge difference is 0 then no waters are needed
    # return early with an empty list
    if charge_difference == 0:
        return []

    # construct a new mdt trajectory
    traj = mdt.Trajectory(
        positions[np.newaxis, ...],
        mdt.Topology.from_openmm(topology)
    )

    water_atoms = traj.topology.select("water")
    solvent_residue_names = list(_SOLVENT_TYPES)
    solute_atoms = [atom.index for atom in traj.topology.atoms
                    if atom.residue.name not in solvent_residue_names]

    excluded_waters = mdt.compute_neighbors(
        traj, distance_cutoff.to(unit.nanometer).m,
        solute_atoms, haystack_indices=water_atoms,
        periodic=True,
    )[0]

    solvent_indices = set([
        atom.residue.index for atom in traj.topology.atoms
        if (atom.index in water_atoms) and (atom.index not in excluded_waters)
    ])

    if len(solvent_indices) < 1:
        errmsg = ("There are no waters outside of a "
                  f"{distance_cutoff.to(unit.nanometer)} nanometer distance "
                  "of the system solutes to be used as alchemical waters")
        raise ValueError(errmsg)

    # unlike the original perses approach, we stick to the first water index
    # in order to make sure we somewhat reproducibily pick the same water
    chosen_residues = list(solvent_indices)[:abs(charge_difference)]

    return chosen_residues


def combined_topology(topology1: app.Topology,
                      topology2: app.Topology,
                      exclude_resids: Optional[npt.NDArray] = None,):
    """
    Create a new topology combining these two topologies.

    The box information from the *first* topology will be copied over

    Parameters
    ----------
    topology1 : openmm.app.Topology
      Topology of the template system to graft topology2 into.
    topology2 : openmm.app.Topology
      Topology to combine (not in place) with topology1.
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
        exclude_resids = np.array([])

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
    resids : npt.NDArrayLike
        An array of residue indices which match the residues we want to get
        atom indices for.
    """
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
    from collections import Counter

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

    # count the number of times each atom appears
    to_del_counts = Counter(to_del)
    # if a H-atom appears more than once, it means it was involved in
    # multiple different constraints at the end states but that the atom is in the core region
    # this should not happen
    for idx, count in to_del_counts.items():
        if count > 1:
            # this is raised before we hit the KeyError below
            raise ValueError(f"Atom {idx} was involved in {count} unique constraints "
                             f" that changed between the two end-states. This should not happen for core "
                             f"atoms, please check your atom mapping. Please raise an issue on the openfe github with "
                             f"the steps to reproduce this error for more help.")
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
              Note: This will not contain the indices of any alchemical waters!
            8. new_mol_indices
              Indices of the alchemical molecule in the new system.
              Note: This will not contain the indices of any alchemical waters!
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


def _create_bond_lookup(htf: HybridTopologyFactory) -> dict:
    """
    Create a lookup dictionary of bonded atoms in the hybrid topology.

    Parameters
    ----------
    htf : DevelopmentHybridTopologyFactory
        The hybrid topology factory containing the hybrid topology.

    Returns
    -------
    dict
        A dictionary where keys are atom indices and values are sets of bonded atom indices.
    """
    # get all the bonds in the hybrid topology so we can check for an improper
    hybrid_bonds = list(htf.omm_hybrid_topology.bonds())
    # construct a lookup of bonded atoms
    bonded_atom_lookup = defaultdict(set)
    for bond in hybrid_bonds:
        a1 = bond[0].index
        a2 = bond[1].index
        bonded_atom_lookup[a1].add(a2)
        bonded_atom_lookup[a2].add(a1)
    return bonded_atom_lookup


def _copy_hybrid_system(htf: HybridTopologyFactory) -> openmm.System:
    """
    Create a deep copy of the given HTF which is ready to be modified by scaling or ghostly corrections.

    Parameters
    ----------
    htf : HybridTopologyFactory
        The hybrid topology factory with the hybrid openmm system to copy.

    Returns
    -------
    openmm.System
        A minimal copy of the hybrid system ready for modification.

    Notes
    -----
    Things copied:
    - hybrid system particles
    - hybrid system constraints
    - barostat if present and box vectors
    """
    new_hybrid_system = openmm.System()
    # add all the particles
    for i in range(htf.hybrid_system.getNumParticles()):
        new_hybrid_system.addParticle(htf.hybrid_system.getParticleMass(i))
    # add all constraints
    for i in range(htf.hybrid_system.getNumConstraints()):
        p1, p2, dist = htf.hybrid_system.getConstraintParameters(i)
        new_hybrid_system.addConstraint(p1, p2, dist)
    # add the barostat and the box vectors
    for force in htf.hybrid_system.getForces():
        if isinstance(force, openmm.MonteCarloBarostat) or isinstance(force, openmm.MonteCarloMembraneBarostat):
            new_hybrid_system.addForce(deepcopy(force))
            # also add the box vectors
            box_vectors = htf.hybrid_system.getDefaultPeriodicBoxVectors()
            new_hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)
            break
    return new_hybrid_system


def _scale_angles_and_torsions(htf: HybridTopologyFactory, scale_factor: float = 0.1, scale_angles: bool = True) -> HybridTopologyFactory:
    """
    Scale all angles and torsion force constants in the dummy-core junction by the given scale factor.

    Parameters
    ----------
    htf : HybridTopologyFactory
        The hybrid topology factory to modify.
    scale_factor : float, optional
        The factor by which to scale the angles and torsion (0 to 1) for constants by for dummy-core junction terms,
        by default 0.1.
    scale_angles : bool, optional
        Whether to scale angles (True) or not (False) in the dummy-core junction, by default True.

    Returns
    -------
    HybridTopologyFactory
        A new HTF with softened angles and torsions crossing the dummy core junctions in the hybrid system.
    """
    assert 0 <= scale_factor <= 1, "Scale factor must be between 0 and 1."

    logger.info(f"Softening angles and torsions involving dummy atoms in the hybrid system by {(1.0 - scale_factor) * 100}%.")
    dummy_old_atoms = htf._atom_classes["unique_old_atoms"]
    dummy_new_atoms = htf._atom_classes["unique_new_atoms"]

    softened_hybrid_system = openmm.System()
    # add all the particles
    logger.info("Copying particles and constraints to new hybrid system.")
    print("Copying particles and constraints to new hybrid system.")
    softened_hybrid_system = _copy_hybrid_system(htf=htf)

    hybrid_forces = htf._hybrid_system_forces
    # copy all forces which do not need to be modified
    # We are only modifying angle and torsion forces which involve the bridge and dummy atoms
    # As the HTF stores all terms involving dummies in the standard forces we can copy all others directly
    # The interpolated forces only contain terms for the core mapped atoms so we don't need to remove any
    forces_not_to_copy = ["standard_angle_force", "unique_atom_torsion_force"]
    if not scale_angles:
        logger.info("Scaling of angles disabled. Only torsions will be softened.")
        forces_not_to_copy.remove("standard_angle_force")
    for force_name, hybrid_force in hybrid_forces.items():
        if force_name not in forces_not_to_copy:
            logger.info(f"Copying force {force_name} to new hybrid system without modification.")
            new_force = deepcopy(hybrid_force)
            softened_hybrid_system.addForce(new_force)

    # now apply the softening to the angle and torsion forces
    # first add a new torsion force to the system
    softened_torsion_force = openmm.PeriodicTorsionForce()
    softened_hybrid_system.addForce(softened_torsion_force)

    # get a quick lookup of the forces
    new_hybrid_forces = {force.getName(): force for force in softened_hybrid_system.getForces()}

    # process angles
    if scale_angles:
        # if we scale angles add a new angle force to the system
        softened_harmonic_angle_force = openmm.HarmonicAngleForce()
        softened_hybrid_system.addForce(softened_harmonic_angle_force)
        logger.info("Processing dummy-core junction angles for softening.")
        print("Processing dummy-core junction angles for softening.")
        logger.info("Adding softened angles to core_angle_force.")
        print("Adding softened angles to core_angle_force.")
        default_hybrid_angle_force = hybrid_forces["standard_angle_force"]
        softened_custom_angle_force = new_hybrid_forces["CustomAngleForce"]
        for i in range(default_hybrid_angle_force.getNumAngles()):
            p1, p2, p3, theta_eq, k = default_hybrid_angle_force.getAngleParameters(i)
            angle = (p1, p2, p3)
            # for the angle terms there must be at least one core atom and 1 or 2 dummy atoms
            # check lambda = 0 first
            if 1 <= len(dummy_new_atoms.intersection(angle)) < 3:
                # if we match a new unique atom the angle must be softened at lambda = 0
                # add the term to the interpolated custom angle force
                new_k = k * scale_factor
                logger.info(f"Softening angle {angle} at lambda=0: original k = {k}, new k = {new_k}")
                print(f"Softening angle {angle} at lambda=0: original k = {k}, new k = {new_k}")
                softened_custom_angle_force.addAngle(p1, p2, p3, [theta_eq, new_k, theta_eq, k])
            elif 1 <= len(dummy_old_atoms.intersection(angle)) < 3:
                # if we match an old unique atom the angle must be softened at lambda = 1
                # add the term to the interpolated custom angle force
                new_k = k * scale_factor
                logger.info(f"Softening angle {angle} at lambda=1: original k = {k}, new k = {new_k}")
                print(f"Softening angle {angle} at lambda=1: original k = {k}, new k = {new_k}")
                softened_custom_angle_force.addAngle(p1, p2, p3, [theta_eq, k, theta_eq, new_k])
            else:
                # the term does not involve any dummy atoms, so we can just copy it
                softened_harmonic_angle_force.addAngle(p1, p2, p3, theta_eq, k)

    # process torsions
    logger.info("Processing dummy-core junction torsions for softening.")
    print("Processing dummy-core junction torsions for softening.")
    logger.info("Adding softened torsions to core_torsion_force.")
    print("Adding softened torsions to core_torsion_force.")
    default_hybrid_torsion_force = hybrid_forces["unique_atom_torsion_force"]
    softened_custom_torsion_force = new_hybrid_forces["CustomTorsionForce"]
    for i in range(default_hybrid_torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = default_hybrid_torsion_force.getTorsionParameters(i)
        torsion = (p1, p2, p3, p4)
        # for the torsion terms there must be at least one core atom and 1-3 dummy atoms
        # check lambda = 0 first
        if 1 <= len(dummy_new_atoms.intersection(torsion)) < 4:
            # if we match a new unique atom the torsion must be softened at lambda = 0
            # add the term to the interpolated custom torsion force
            new_k = k * scale_factor
            logger.info(f"Softening torsion {torsion} at lambda=0: original k = {k}, new k = {new_k}")
            print(f"Softening torsion {torsion} at lambda=0: original k = {k}, new k = {new_k}")
            softened_custom_torsion_force.addTorsion(p1, p2, p3, p4,
                                                     [periodicity, phase,
                                            new_k, periodicity,
                                             phase, k])
        elif 1 <= len(dummy_old_atoms.intersection(torsion)) < 4:
            # if we match an old unique atom the torsion must be softened at lambda = 1
            # add the term to the interpolated custom torsion force
            new_k = k * scale_factor
            logger.info(f"Softening torsion {torsion} at lambda=1: original k = {k}, new k = {new_k}")
            print(f"Softening torsion {torsion} at lambda=1: original k = {k}, new k = {new_k}")
            softened_custom_torsion_force.addTorsion(p1, p2, p3, p4,
                                                     [periodicity, phase,
                                            k, periodicity,
                                             phase, new_k])
        else:
            # the term does not involve any dummy atoms, so we can just copy it
            softened_torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)

    htf._hybrid_system = softened_hybrid_system
    # set the hybrid system forces dict to the new one
    htf._hybrid_system_forces = {force.getName(): force for force in softened_hybrid_system.getForces()}
    return htf


def _load_ghostly_corrections(ghostly_output_string: str) -> dict:
    """
    Parse the Ghostly modification output JSON string and return the corrections as a dictionary.

    Notes
    -----
    - The corrections are returned in a dictionary with the same structure as the Ghostly output but we use sets for the removed and stiffened angles/dihedrals for faster lookup.
    """

    corrections = json.loads(ghostly_output_string)
    # convert the string keys back to tuples
    for lambda_key in corrections.keys():
        for correction_type in corrections[lambda_key].keys():
            # these are stings for some reason, convert them back to tuples
            if correction_type in ["removed_angles", "removed_dihedrals"]:
                corrections[lambda_key][correction_type] = set([ast.literal_eval(tup_str) for tup_str in corrections[lambda_key][correction_type]])
                # this is a list so we need to convert each to a tuple
            elif correction_type == "stiffened_angles":
                corrections[lambda_key][correction_type] = set([tuple(angle) for angle in corrections[lambda_key][correction_type]])
            elif correction_type == "softened_angles":
                new_dict = {}
                for tup_str, params in corrections[lambda_key][correction_type].items():
                    tup = ast.literal_eval(tup_str)
                    new_dict[tup] = params
                corrections[lambda_key][correction_type] = new_dict
    return corrections


def _shift_ghostly_correction_indices(htf: HybridTopologyFactory, corrections: dict) -> dict:
    """Shift the atom indices in the ghostly corrections to account for the solvent environment in the HTF."""
    core_atoms = htf._atom_classes["core_atoms"]
    unique_old_atoms = htf._atom_classes["unique_old_atoms"]
    # get the maximum index of the core and unique old atoms all stateB atoms are after this
    max_state_a_atom = max(core_atoms.union(unique_old_atoms))


    # we need the shift the indices of the atoms in the corrections to account for the solvent system
    # this involves shifting the unique new atoms to be after the last water atom
    # only needed if not in the vacuum leg
    if len(htf._atom_classes["environment_atoms"]) > 0:
        print("shifting ghostly correction atom indices to account for solvent environment.")
        last_water_atom = max(htf._atom_classes["environment_atoms"])
        print("last water atom", last_water_atom)
        last_core_atom = max(htf._atom_classes["core_atoms"])
        print("last core atom", last_core_atom)
        print("old unique atoms", htf._atom_classes["unique_old_atoms"])
        print("new unique atoms", htf._atom_classes["unique_new_atoms"])
        shifted_corrections = {}
        for lambda_key in corrections.keys():
            for correction_type in corrections[lambda_key].keys():
                if correction_type in ["removed_angles", "removed_dihedrals", "stiffened_angles"]:
                    shifted_list = set()
                    for angle in corrections[lambda_key][correction_type]:
                        shifted_angle = tuple(
                            atom_idx + (last_water_atom - max_state_a_atom)
                            # only shift if not a core atom and greater than last core atom
                            # else this is a unique old atom and should not be shifted
                            if atom_idx not in core_atoms and atom_idx > max_state_a_atom else atom_idx
                            for atom_idx in angle
                        )
                        shifted_list.add(shifted_angle)
                    shifted_corrections.setdefault(lambda_key, {})[correction_type] = shifted_list
                elif correction_type == "softened_angles":
                    new_dict = {}
                    for angle, params in corrections[lambda_key][correction_type].items():
                        shifted_angle = tuple(
                            atom_idx + (last_water_atom - max_state_a_atom)
                            if atom_idx not in core_atoms and atom_idx > max_state_a_atom else atom_idx
                            for atom_idx in angle
                        )
                        new_dict[shifted_angle] = params
                    shifted_corrections.setdefault(lambda_key, {})[correction_type] = new_dict
        corrections = shifted_corrections
        print("new corrections", corrections)
    return corrections


def _apply_ghostly_corrections(htf: HybridTopologyFactory, corrections: dict) -> HybridTopologyFactory:
    """
    Apply the ghostly corrections parsed from the output file to the HTF.

    Notes
    -----
    - The HTF is edited inplace due to issues with deepcopying the HTF object.
    - The method will track which corrections were applied and compare them to the supplied corrections.
    - The method will check that a correction is applied to all junctions involving dummy atoms identified using an internal method.

    Raises
    ------
    AssertionError
        If a parameter is changed by ghostly but we can not determine what type of correction it was.
    ValueError
        If a correction provided by ghostly is not applied to the HTF.
    """
    logger.info("Applying ghostly corrections to hybrid system.")
    dummy_old_atoms = htf._atom_classes["unique_old_atoms"]
    dummy_new_atoms = htf._atom_classes["unique_new_atoms"]

    corrections = _shift_ghostly_correction_indices(htf, corrections)

    new_hybrid_system = _copy_hybrid_system(htf=htf)

    hybrid_forces = htf._hybrid_system_forces
    # copy all forces which do not need to be modified
    # We are only modifying angle and torsion forces with ghostly corrections
    # As the HTF stores all terms involving ghosts in the standard forces we can copy all others directly
    # The interpolated forces only contain terms for the core mapped atoms so we don't need to remove any
    forces_not_to_copy = ["standard_angle_force", "unique_atom_torsion_force"]
    for force_name, hybrid_force in hybrid_forces.items():
        if force_name not in forces_not_to_copy:
            new_force = deepcopy(hybrid_force)
            new_hybrid_system.addForce(new_force)

    # now apply the ghostly corrections to the angle and torsion forces
    # first add a new standard angle and torsion force to the system
    new_harmonic_angle_force = openmm.HarmonicAngleForce()
    new_hybrid_system.addForce(new_harmonic_angle_force)
    new_torsion_force = openmm.PeriodicTorsionForce()
    new_hybrid_system.addForce(new_torsion_force)
    # get a quick lookup of the forces
    new_hybrid_forces = {force.getName(): force for force in new_hybrid_system.getForces()}

    # track the applied corrections
    applied_corrections = {
        "lambda_0": {"removed_angles": set(), "stiffened_angles": set(), "softened_angles": set(), "removed_dihedrals": set()},
        "lambda_1": {"removed_angles": set(), "stiffened_angles": set(), "softened_angles": set(), "removed_dihedrals": set()}
       }

    # process angles
    custom_angle_force = new_hybrid_forces["CustomAngleForce"]
    old_hybrid_angle_force = hybrid_forces["standard_angle_force"]

    # set up the angle parameters for stiffening and zeroing
    ZERO_K = 0.0 * omm_unit.kilocalories_per_mole / (omm_unit.radian ** 2)
    STIFF_K = 100.0 * omm_unit.kilocalories_per_mole / (omm_unit.radian ** 2)
    STIFF_THETA = 0.5 * math.pi * omm_unit.radian

    for angle_idx in range(old_hybrid_angle_force.getNumAngles()):
        p1, p2, p3, theta_eq, k = old_hybrid_angle_force.getAngleParameters(angle_idx)
        # check if we have one ghost atom for this angle
        angle = (p1, p2, p3)
        if 1 <= len(dummy_old_atoms.intersection(angle)) < 3 or 1<= len(dummy_new_atoms.intersection(angle)) < 3:
            angle_reversed = (p3, p2, p1)
            # set up containers for the end state values
            lambda_0_k = k
            lambda_0_theta_eq = theta_eq
            lambda_1_k = k
            lambda_1_theta_eq = theta_eq
            end_state, correction_type = None, None

            # check for removed angles
            if (prob_angle:= angle) in corrections["lambda_0"]["removed_angles"] or (prob_angle:= angle_reversed) in corrections["lambda_0"]["removed_angles"]:
                lambda_0_k = ZERO_K
                end_state = 0
                correction_type = "removed_angles"
            elif (prob_angle:= angle) in corrections["lambda_1"]["removed_angles"] or (prob_angle:= angle_reversed) in corrections["lambda_1"]["removed_angles"]:
                lambda_1_k = ZERO_K
                end_state = 1
                correction_type = "removed_angles"
            # check for stiffened angles
            elif (prob_angle:= angle) in corrections["lambda_0"]["stiffened_angles"] or (prob_angle:= angle_reversed) in corrections["lambda_0"]["stiffened_angles"]:
                lambda_0_k = STIFF_K  # default stiffening k value
                lambda_0_theta_eq = STIFF_THETA  # 90 degrees
                end_state = 0
                correction_type = "stiffened_angles"
            elif (prob_angle:= angle) in corrections["lambda_1"]["stiffened_angles"] or (prob_angle:= angle_reversed) in corrections["lambda_1"]["stiffened_angles"]:
                lambda_1_k = STIFF_K  # default stiffening k value
                lambda_1_theta_eq = STIFF_THETA  # 90 degrees
                end_state = 1
                correction_type = "stiffened_angles"
                # check for softened angles
            elif (prob_angle:= angle) in corrections["lambda_0"]["softened_angles"] or (prob_angle:= angle_reversed) in corrections["lambda_0"]["softened_angles"]:
                soften_params = corrections["lambda_0"]["softened_angles"][prob_angle]
                lambda_0_k = soften_params["k"] * omm_unit.kilocalories_per_mole / (omm_unit.radian ** 2)
                lambda_0_theta_eq = soften_params["theta0"] * omm_unit.radian
                end_state = 0
                correction_type = "softened_angles"
            elif (prob_angle:= angle) in corrections["lambda_1"]["softened_angles"] or (prob_angle:= angle_reversed) in corrections["lambda_1"]["softened_angles"]:
                soften_params = corrections["lambda_1"]["softened_angles"][prob_angle]
                lambda_1_k = soften_params["k"] * omm_unit.kilocalories_per_mole / (omm_unit.radian ** 2)
                lambda_1_theta_eq = soften_params["theta0"] * omm_unit.radian
                end_state = 1
                correction_type = "softened_angles"

            # some angles involving dummy atoms need to be kept to ensure 3 redundant connections
            if lambda_0_k != lambda_1_k or lambda_0_theta_eq != lambda_1_theta_eq:
                # add the term to the interpolated custom angle force
                print(f"Applying ghostly angle correction for angle {angle}: "
                      f"lambda_0 k = {lambda_0_k}, theta_eq = {lambda_0_theta_eq}; "
                      f"lambda_1 k = {lambda_1_k}, theta_eq = {lambda_1_theta_eq}")
                logger.info(f"Applying ghostly angle correction for angle {angle}: "
                      f"lambda_0 k = {lambda_0_k}, theta_eq = {lambda_0_theta_eq}; "
                      f"lambda_1 k = {lambda_1_k}, theta_eq = {lambda_1_theta_eq}")
                custom_angle_force.addAngle(p1, p2, p3,
                                            [lambda_0_theta_eq, lambda_0_k,
                                            lambda_1_theta_eq, lambda_1_k])
                # log this as a correction applied
                assert correction_type is not None, "Correction type should not be None if k or theta_eq differ!"
                applied_corrections[f"lambda_{end_state}"][correction_type].add(prob_angle)

            else:
                # both k and theta_eq values are the same, just add to the standard angle force
                new_harmonic_angle_force.addAngle(p1, p2, p3, theta_eq, k)

        else:
            # the term does not involve any ghost atoms, so we can just copy it
            new_harmonic_angle_force.addAngle(p1, p2, p3, theta_eq, k)

    # process torsions
    custom_torsion_force = new_hybrid_forces["CustomTorsionForce"]
    old_hybrid_torsion_force = hybrid_forces["unique_atom_torsion_force"]

    # set up the torsion parameters for zeroing
    TORSION_ZERO_K = 0.0 * omm_unit.kilocalories_per_mole

    # get all the bonds in the hybrid topology so we can check for an improper
    bonded_atom_lookup = _create_bond_lookup(htf=htf)

    for torsion_idx in range(old_hybrid_torsion_force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = old_hybrid_torsion_force.getTorsionParameters(torsion_idx)
        # check if we have one ghost atom for this torsion
        torsion = (p1, p2, p3, p4)
        if 1<= len(dummy_old_atoms.intersection(torsion)) < 4 or 1<= len(dummy_new_atoms.intersection(torsion)) < 4:
            torsion_reversed = (p4, p3, p2, p1)
            # check if we have an improper torsion (central atoms bonded)
            if not (p1 in bonded_atom_lookup[p2] and p2 in bonded_atom_lookup[p3] and p3 in bonded_atom_lookup[p4]):
                # this is an improper with a dummy atom and should be skipped
                # generate all permutations of the other atoms to check for removal
                central_atom = p1
                other_atoms = [p2, p3, p4]
                torsion_variants = set()
                for perm in itertools.permutations(other_atoms):
                    torsion_variants.add((central_atom, perm[0], perm[1], perm[2]))
                    # add the reverse as well as ghostly may list either
                    torsion_variants.add((perm[2], perm[1], perm[0], central_atom))
            else:
                torsion_variants = {torsion, torsion_reversed}

            # set up containers for the end state values
            lambda_0_k = k
            lambda_1_k = k
            end_state = None

            # check for removed dihedrals
            if matched:= corrections["lambda_0"]["removed_dihedrals"].intersection(torsion_variants):
                lambda_0_k = TORSION_ZERO_K
                end_state = 0
            elif matched:= corrections["lambda_1"]["removed_dihedrals"].intersection(torsion_variants):
                lambda_1_k = TORSION_ZERO_K
                end_state = 1
            # some dihedrals involving ghost atoms need to be kept to ensure 3 redundant connections
            if lambda_0_k != lambda_1_k:
                # add the term to the interpolated custom torsion force
                print(f"Applying ghostly torsion correction for torsion {torsion}: "
                      f"lambda_0 k = {lambda_0_k}; "
                      f"lambda_1 k = {lambda_1_k}")
                logger.info(f"Applying ghostly torsion correction for torsion {torsion}: "
                      f"lambda_0 k = {lambda_0_k}; "
                      f"lambda_1 k = {lambda_1_k}")
                custom_torsion_force.addTorsion(p1, p2, p3, p4,
                                                [periodicity, phase,
                                                lambda_0_k, periodicity,
                                                 phase, lambda_1_k])
                # log this as a correction applied
                assert end_state is not None, "End state should not be None if k values differ!"
                applied_corrections[f"lambda_{end_state}"]["removed_dihedrals"].update(matched)
            else:
                # both k values are the same, just add to the standard torsion force
                new_torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)
        else:
            # the term does not involve any ghost atoms, so we can just copy it
            new_torsion_force.addTorsion(p1, p2, p3, p4, periodicity, phase, k)


    # compare the supplied and applied corrections
    for lambda_key in corrections.keys():
        for correction_type in corrections[lambda_key].keys():
            supplied = set()
            applied = applied_corrections[lambda_key][correction_type]
            if correction_type in ["removed_angles", "stiffened_angles"]:
                supplied = corrections[lambda_key][correction_type]
            elif correction_type == "softened_angles":
                supplied = set([tuple(tup) for tup in corrections[lambda_key][correction_type].keys()])
            elif correction_type == "removed_dihedrals":
                supplied = corrections[lambda_key][correction_type]
            not_applied = supplied - applied
            if len(not_applied) > 0 and correction_type == "removed_dihedrals":
                # in some cases dihedrals are listed to be removed but are not present in the HTF these involve linear nitrile groups for example
                # check if these missed dihedrals are this type
                dummy_group = dummy_new_atoms if lambda_key == "lambda_1" else dummy_old_atoms
                for dihedral in list(not_applied):
                    # if this is a linear group torsion then atom 2 or 3 will be bonded to only 2 other atoms
                    a2_bonds = bonded_atom_lookup[dihedral[1]] - dummy_group
                    a3_bonds = bonded_atom_lookup[dihedral[2]] - dummy_group
                    if len(a2_bonds) <= 2 or len(a3_bonds) <= 2:
                        not_applied.remove(dihedral)
            if len(not_applied) > 0:
                raise ValueError(f"The following {correction_type} corrections for {lambda_key} were not applied: {not_applied}")


    htf._hybrid_system = new_hybrid_system
    # set the hybrid system forces dict to the new one
    htf._hybrid_system_forces = {force.getName(): force for force in new_hybrid_system.getForces()}
    return htf