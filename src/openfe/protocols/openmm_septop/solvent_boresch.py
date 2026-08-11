# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""
Solvent-leg Boresch restraint wiring for SepTopSolventSetupUnit.

This module provides a drop-in replacement for the ``_add_restraints`` method
of ``SepTopSolventSetupUnit``.  It replaces the single harmonic distance
restraint with two independent Boresch restraints (one per ligand), each
anchored to three dedicated dummy atoms.

Lambda schedule
---------------
The restraints use controlling parameter names
``"lambda_restraints_A"`` and ``"lambda_restraints_B"``.  For the solvent
leg, both should remain at 1.0 throughout all lambda windows (i.e. the
restraints are always on).

Standard state correction
-------------------------
For the solvent leg the Boresch correction has the *same* sign for both
ligands — they are both being restrained (neither is being released).
The net contribution to the RBFE therefore cancels when taking the
difference. We return it for bookkeeping but it should not be applied
asymmetrically as in the complex leg.
"""
from __future__ import annotations

import numpy as np
import openmm
import openmm.unit as omm_unit
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from gufe.settings.models import SettingsBaseModel
from openff.units import Quantity
from openff.units.openmm import to_openmm
from openmmtools.states import ThermodynamicState
from rdkit import Chem

from openfe.protocols.restraint_utils import geometry
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.protocols.restraint_utils.geometry.boresch import find_guest_atom_candidates
from openfe.protocols.restraint_utils.openmm.omm_restraints import (
    BoreschRestraint,
    add_force_in_separate_group,
)
from openfe.protocols.restraint_utils.settings import BoreschRestraintSettings

from openfe.protocols.restraint_utils.geometry.boresch.dummy import find_dummy_atom_positions
from openfe.protocols.restraint_utils.openmm.omm_dummy import add_dummy_atoms_to_system


def add_solvent_boresch_restraints(
    system: openmm.System,
    positions_ang: np.ndarray,
    rdmol_A: Chem.Mol,
    rdmol_B: Chem.Mol,
    ligand_A_idxs: list[int],
    ligand_B_idxs: list[int],
    settings: dict[str, SettingsBaseModel],
) -> tuple[
    Quantity,
    Quantity,
    openmm.System,
    np.ndarray,
    BoreschRestraintGeometry,
    BoreschRestraintGeometry,
]:
    """
    Add Boresch restraints for both ligands to the solvent-leg System,
    using analytically-placed dummy atoms as hosts.

    Parameters
    ----------
    system:
        The (alchemical) OpenMM System to modify. **Modified in-place.**
    positions_ang:
        Full-system positions in Angstroms, shape ``(N, 3)``. Must already
        reflect the desired input conformer positions for both ligands.
    rdmol_A, rdmol_B:
        Sanitised RDKit molecules for ligands A and B.
    ligand_A_idxs, ligand_B_idxs:
        Atom indices for each ligand in the full system.
    settings:
        The protocol settings dict, must contain:
        * ``"restraint_settings"``  — a ``BoreschRestraintSettings`` instance
        * ``"thermo_settings"``     — for temperature / pressure

    Returns
    -------
    correction_A:
        Boresch standard-state correction for ligand A (kJ/mol).
    correction_B:
        Boresch standard-state correction for ligand B (kJ/mol).
    system:
        The modified System (same object, returned for convenience).
    positions_ang:
        Extended positions array with dummy atom coordinates appended,
        shape ``(N + 6, 3)``.
    geom_A:
        The BoreschRestraintGeometry applied to ligand A (host_atoms are
        the 3 dummy atom indices for ligand A; guest_atoms are G0/G1/G2).
    geom_B:
        The BoreschRestraintGeometry applied to ligand B.

    Raises
    ------
    TypeError
        If ``settings["restraint_settings"]`` is not a
        ``BoreschRestraintSettings`` instance.
    ValueError
        If no suitable ligand anchor atoms can be found for either ligand.
    """
    restraint_settings = settings["restraint_settings"]
    if not isinstance(restraint_settings, BoreschRestraintSettings):
        raise TypeError(
            "Solvent-leg dummy Boresch restraints require a "
            f"BoreschRestraintSettings instance, got "
            f"{type(restraint_settings).__name__}."
        )

    # 1. Add 6 dummy atoms to the system (3 per ligand).
    #    Positions are initialised to zero; we fill them in below.
    system, positions_ang, dummy_idxs_A = add_dummy_atoms_to_system(
        system, positions_ang, n_dummies=3
    )
    system, positions_ang, dummy_idxs_B = add_dummy_atoms_to_system(
        system, positions_ang, n_dummies=3
    )

    # 2. For each ligand: find guest anchor atoms (G0/G1/G2), place the
    #    3 dummy atoms analytically around them, then build the
    #    BoreschRestraintGeometry by delegating to the same
    #    find_boresch_restraint entry point used by the complex leg,
    #    via its guest_restraint_atoms_idxs / host_restraint_atoms_idxs
    #    override path (we already know exactly which atoms to use,
    #    so we skip its host/guest search logic).
    def _build_geometry_for_ligand(
        rdmol: Chem.Mol,
        lig_idxs: list[int],
        dummy_idxs: list[int],
    ) -> BoreschRestraintGeometry:
        """
        Find guest anchor atoms, place dummies, write dummy positions into
        positions_ang in-place, and return a BoreschRestraintGeometry.
        """
        n_atoms = positions_ang.shape[0]
        u = mda.Universe.empty(n_atoms, trajectory=True)
        u.load_new(positions_ang[np.newaxis, :, :], format=MemoryReader)

        anchors = find_guest_atom_candidates(
            universe=u,
            rdmol=rdmol,
            guest_idxs=lig_idxs,
            rmsf_cutoff=restraint_settings.rmsf_cutoff,
        )
        if not anchors:
            raise ValueError(
                "No suitable ligand anchor atoms found for dummy Boresch "
                "restraint. Try using a ligand with aromatic rings or more "
                "rigid heavy atoms."
            )
        g0_idx, g1_idx, g2_idx = anchors[0]

        # Place the 3 dummy atoms analytically and write their positions
        # into positions_ang (and the live MDA universe) before measuring
        # the restraint geometry.
        p_d0, p_d1, p_d2 = find_dummy_atom_positions(
            positions_ang[g0_idx],
            positions_ang[g1_idx],
            positions_ang[g2_idx],
        )
        positions_ang[dummy_idxs[0]] = p_d0
        positions_ang[dummy_idxs[1]] = p_d1
        positions_ang[dummy_idxs[2]] = p_d2
        u.atoms.positions = positions_ang

        # guest_restraint_atoms_idxs are positions *within* lig_idxs, not
        # absolute system indices (mirrors how find_boresch_restraint's
        # override path indexes into universe.atoms[guest_idxs]).
        guest_restraint_atoms_idxs = [
            lig_idxs.index(g0_idx),
            lig_idxs.index(g1_idx),
            lig_idxs.index(g2_idx),
        ]
        # host_restraint_atoms_idxs select all 3 dummies, in placement
        # order (D0, D1, D2), since dummy_idxs is exactly the host pool.
        host_restraint_atoms_idxs = [0, 1, 2]

        return geometry.boresch.find_boresch_restraint(
            universe=u,
            guest_rdmol=rdmol,
            guest_idxs=lig_idxs,
            host_idxs=dummy_idxs,
            guest_restraint_atoms_idxs=guest_restraint_atoms_idxs,
            host_restraint_atoms_idxs=host_restraint_atoms_idxs,
        )

    geom_A = _build_geometry_for_ligand(rdmol_A, ligand_A_idxs, dummy_idxs_A)
    geom_B = _build_geometry_for_ligand(rdmol_B, ligand_B_idxs, dummy_idxs_B)

    # 4. Add the Boresch forces via the existing BoreschRestraint class.
    restraint_A = BoreschRestraint(restraint_settings)
    restraint_B = BoreschRestraint(restraint_settings)

    thermodynamic_state = ThermodynamicState(
        system,
        temperature=to_openmm(settings["thermo_settings"].temperature),
        pressure=to_openmm(settings["thermo_settings"].pressure),
    )

    restraint_A.add_force(
        thermodynamic_state,
        geom_A,
        controlling_parameter_name="lambda_restraints_A",
    )
    restraint_B.add_force(
        thermodynamic_state,
        geom_B,
        controlling_parameter_name="lambda_restraints_B",
    )

    # 5. Standard state corrections.
    #    In the solvent leg both ligands are restrained throughout, so
    #    corrections are equal in magnitude. We return them for
    #    bookkeeping; they cancel in the RBFE cycle.
    correction_A = restraint_A.get_standard_state_correction(
        thermodynamic_state, geom_A
    )
    correction_B = restraint_B.get_standard_state_correction(
        thermodynamic_state, geom_B
    )

    system = thermodynamic_state.get_system(remove_thermostat=True)

    return correction_A, correction_B, system, positions_ang, geom_A, geom_B