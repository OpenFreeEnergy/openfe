# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Reusable unclassified utility methods for OpenMM-based alchemical
Protocols.
"""
from pathlib import Path
from numpy import typing as npt
from openmm import app


def subsample_omm_topology(
    topology: app.Topology,
    subsample_indices: tuple,
) -> app.Topology:
    """
    Taking an initial app.Topology and a tuple of atom indices, generate
    a new app.Topology which only contains those atoms and any bonds which
    are involved between these atoms.

    Parameters
    ----------
    topology : app.Topology
      Initial Topology to subsample from
    subsample_indices : tuple
      Tuple of atom indices for the atoms we want to subsample from the
      input Topology.

    Returns
    -------
    top : app.Topology
      Subsampled Topology.
    """
    top = app.Topology()
    chains = set()
    residues = set()
    old_to_new_atom_map = {}

    # First pass - get chain & resid info
    for at in topology.atoms():
        if at.index in subsample_indices:
            residues.add(at.residue.index)
            chains.add(at.residue.chain.index)

    # Now let's actually add things
    for chain_id, chain in enumerate(topology.chains()):
        if chain.index in chains:
            new_chain = top.addChain(chain_id)

            for res in chain.residues():
                if res.index in residues:
                    new_res = top.addResidue(res.name, new_chain, res.id)

                    for at in res.atoms():
                        if at.index in subsample_indices:
                            new_atom = top.addAtom(at.name, at.element,
                                                   new_res, at.id)
                            # I think MDTraj is doing bad things to bond
                            # objects, so the equality isn't working. Using
                            # indices to get around it
                            old_to_new_atom_map[at.index] = new_atom

    for bond in topology.bonds():
        if bond.atom1.index in subsample_indices:
            if bond.atom2.index in subsample_indices:
                top.addBond(old_to_new_atom_map[bond.atom1.index],
                            old_to_new_atom_map[bond.atom2.index],
                            bond.type, bond.order)

    return top


def write_subsampled_pdb(
    topology: app.Topology,
    positions: npt.NDArray,
    subsample_indices: tuple,
    outfile: Path,
) -> None:
    """
    Write out a PDB of the subsampled system.

    Parameters
    ----------
    topology : app.Topology
      Initial Topology to subsample from.
    positions: npt.NDArray
      Positions of the system to subsample from.
    subsample_indices : tuple
      Tuple of atom indices for the atoms we want to subsample from the
      input Topology.
    outfile : pathlib.Path
      File to write PDB to.
    """
    # Get a subsampled topology
    sub_top = subsample_omm_topology(
        topology, subsample_indices
    )

    # Subsample the positions
    sub_pos = positions[subsample_indices, :]


    # Write it out
    with open(outfile, 'w') as f:
        app.PDBFile.writeFile(sub_top, sub_pos, f)
