from itertools import chain

import numpy as np
import openmm
from gufe import LigandAtomMapping, ProteinComponent, SolventComponent
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import app, unit
from openmmforcefields.generators import SystemGenerator

from openfe.protocols.openmm_rfe import _rfe_utils
from openfe.protocols.openmm_rfe._rfe_utils.relative import HybridTopologyFactory
from openfe.protocols.openmm_utils import system_creation


def make_htf(
    mapping: LigandAtomMapping,
    settings,
    protein: ProteinComponent = None,
    solvent: SolventComponent = None,
) -> HybridTopologyFactory:
    """Code copied from the RBFE protocol to make an HTF."""

    system_generator = SystemGenerator(
        forcefields=settings.forcefield_settings.forcefields,
        small_molecule_forcefield=settings.forcefield_settings.small_molecule_forcefield,
        forcefield_kwargs={
            "constraints": app.HBonds,
            "rigidWater": True,
            "hydrogenMass": settings.forcefield_settings.hydrogen_mass * unit.amu,
            "removeCMMotion": settings.integrator_settings.remove_com,
        },
        periodic_forcefield_kwargs={
            "nonbondedMethod": app.PME,
            "nonbondedCutoff": 0.9 * unit.nanometers,
        },
        barostat=openmm.MonteCarloBarostat(
            ensure_quantity(settings.thermo_settings.pressure, "openmm"),
            ensure_quantity(settings.thermo_settings.temperature, "openmm"),
            settings.integrator_settings.barostat_frequency.m,
        ),
        cache=None,
    )
    small_mols = [mapping.componentA, mapping.componentB]
    # copy a lot of code from the RHT protocol
    off_small_mols = {
        "stateA": [(mapping.componentA, mapping.componentA.to_openff())],
        "stateB": [(mapping.componentB, mapping.componentB.to_openff())],
        "both": [
            (m, m.to_openff())
            for m in small_mols
            if (m != mapping.componentA and m != mapping.componentB)
        ],
    }

    # c. force the creation of parameters
    # This is necessary because we need to have the FF templates
    # registered ahead of solvating the system.
    for smc, mol in chain(
        off_small_mols["stateA"], off_small_mols["stateB"], off_small_mols["both"]
    ):
        system_generator.create_system(mol.to_topology().to_openmm(), molecules=[mol])

    # c. get OpenMM Modeller + a dictionary of resids for each component
    stateA_modeller, comp_resids = system_creation.get_omm_modeller(
        # add the protein if passed
        protein_comp=protein,
        # add the solvent if passed
        solvent_comp=solvent,
        small_mols=dict(chain(off_small_mols["stateA"], off_small_mols["both"])),
        omm_forcefield=system_generator.forcefield,
        solvent_settings=settings.solvation_settings,
    )
    # d. get topology & positions
    # Note: roundtrip positions to remove vec3 issues
    stateA_topology = stateA_modeller.getTopology()
    stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

    # e. create the stateA System
    # Block out oechem backend in system_generator calls to avoid
    # any issues with smiles roundtripping between rdkit and oechem
    stateA_system = system_generator.create_system(
        stateA_modeller.topology,
        molecules=[m for _, m in chain(off_small_mols["stateA"], off_small_mols["both"])],
    )

    # 2. Get stateB system
    # a. get the topology
    stateB_topology, stateB_alchem_resids = _rfe_utils.topologyhelpers.combined_topology(
        stateA_topology,
        # zeroth item (there's only one) then get the OFF representation
        off_small_mols["stateB"][0][1].to_topology().to_openmm(),
        exclude_resids=comp_resids[mapping.componentA],
    )

    # b. get a list of small molecules for stateB
    # Block out oechem backend in system_generator calls to avoid
    stateB_system = system_generator.create_system(
        stateB_topology,
        molecules=[m for _, m in chain(off_small_mols["stateB"], off_small_mols["both"])],
    )

    #  c. Define correspondence mappings between the two systems
    ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
        mapping.componentA_to_componentB,
        stateA_system,
        stateA_topology,
        comp_resids[mapping.componentA],
        stateB_system,
        stateB_topology,
        stateB_alchem_resids,
        # These are non-optional settings for this method
        fix_constraints=True,
    )

    #  e. Finally get the positions
    stateB_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
        ligand_mappings,
        stateA_topology,
        stateB_topology,
        old_positions=ensure_quantity(stateA_positions, "openmm"),
        insert_positions=ensure_quantity(off_small_mols["stateB"][0][1].conformers[0], "openmm"),
    )
    return HybridTopologyFactory(
        old_system=stateA_system,
        old_positions=stateA_positions,
        old_topology=stateA_topology,
        new_system=stateB_system,
        new_positions=stateB_positions,
        new_topology=stateB_topology,
        old_to_new_atom_map=ligand_mappings["old_to_new_atom_map"],
        old_to_new_core_atom_map=ligand_mappings["old_to_new_core_atom_map"],
        use_dispersion_correction=settings.alchemical_settings.use_dispersion_correction,
        softcore_alpha=settings.alchemical_settings.softcore_alpha,
        softcore_LJ_v2=True,
        softcore_LJ_v2_alpha=settings.alchemical_settings.softcore_alpha,
        interpolate_old_and_new_14s=settings.alchemical_settings.turn_off_core_unique_exceptions,
    )


def _make_system_with_cmap(
    map_sizes: list[int],
    mapped_torsions: list[tuple[int, int, int, int, int, int, int, int, int]] | None = None,
    num_atoms: int = 8,
):
    """
    Build an OpenMM System with a CMAP term based on the provided mapping data.
    :param map_sizes: Mapping data:
    :param mapped_torsions:
    :param num_atoms:
    :return:
    """
    assert num_atoms >= 8, "num_atoms must be at least 8 to accommodate mapped torsions"
    system = openmm.System()
    # add dummy forces to avoid errors
    for force in [
        openmm.NonbondedForce,
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
    ]:
        system.addForce(force())

    for _ in range(num_atoms):
        system.addParticle(12.0)  # Add carbon-like particles

    # create a CMAP force
    cmap_force = openmm.CMAPTorsionForce()

    for map_size in map_sizes:
        # Create a grid for the CMAP
        grid = [0.0] * (map_size * map_size)
        cmap_force.addMap(map_size, grid)

    if mapped_torsions is None:
        # add a single cmap term for all atoms using the first map
        mapped_torsions = [(0, 0, 1, 2, 3, 4, 5, 6, 7)]

    for torsion in mapped_torsions:
        cmap_force.addTorsion(torsion[0], *torsion[1:])

    system.addForce(cmap_force)
    # build a basic topology for the number of atoms bonding each atom to the next
    topology = openmm.app.Topology()
    chain = topology.addChain()
    res = topology.addResidue("RES", chain)
    atoms = []
    for i in range(num_atoms):
        atom = topology.addAtom(f"C{i + 1}", openmm.app.element.carbon, res)
        atoms.append(atom)
        if i > 0:
            topology.addBond(atoms[i - 1], atoms[i])
    # build a fake set of positions
    positions = openmm.unit.Quantity(np.zeros((num_atoms, 3)), openmm.unit.nanometer)
    return system, topology, positions
