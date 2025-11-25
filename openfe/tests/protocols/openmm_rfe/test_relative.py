import pytest
from openfe.tests.helpers import _make_system_with_cmap
from openfe.protocols.openmm_rfe._rfe_utils.relative import HybridTopologyFactory
from openmm import app, unit
import openmm
from openfe.protocols.openmm_rfe import _rfe_utils
import copy


def test_cmap_system_no_dummy_pme_energy(htf_cmap_chlorobenzene_to_fluorobenzene):
    """
    Test that we can make a hybrid topology for a system with conserved CMAP terms not in the alchemical region and that
    the hybrid energy matches the end state energy.
    """
    htf = htf_cmap_chlorobenzene_to_fluorobenzene["htf"]
    # make sure the cmap force was added to the internal store
    assert "cmap_torsion_force" in htf._hybrid_system_forces
    hybrid_system = htf.hybrid_system
    # make sure we can find the force in the system
    forces = htf_cmap_chlorobenzene_to_fluorobenzene["forces"]
    assert isinstance(forces["CMAPTorsionForce"], openmm.CMAPTorsionForce)

    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    default_lambda = _rfe_utils.lambdaprotocol.LambdaProtocol()

    hybrid_simulation = app.Simulation(
        topology=htf.omm_hybrid_topology,
        system=hybrid_system,
        integrator=integrator,
        platform=platform
    )
    for end_state, ref_system, ref_top, pos in [
        (0, htf._old_system, htf._old_topology, htf._old_positions),
        (1, htf._new_system, htf._new_topology, htf._new_positions)
    ]:
        # set lambda
        # set all lambda values to the current end state
        for name, func in default_lambda.functions.items():
            val = func(end_state)
            hybrid_simulation.context.setParameter(name, val)
        # set positions
        hybrid_simulation.context.setPositions(pos)
        # get the hybrid system energy
        hybrid_state = hybrid_simulation.context.getState(getEnergy=True)
        hybrid_energy = hybrid_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # now create a reference simulation
        ref_simulation = app.Simulation(
            topology=ref_top,
            system=ref_system,
            integrator=copy.deepcopy(integrator),
            platform=platform
        )
        ref_simulation.context.setPositions(pos)
        ref_state = ref_simulation.context.getState(getEnergy=True)
        ref_energy = ref_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # energies should be the same
        assert ref_energy == pytest.approx(hybrid_energy, rel=1e-5)
        # make sure the energy is non-zero to avoid false positives
        assert 0.0 != pytest.approx(hybrid_energy)


def test_verify_cmap_no_cmap():
    """Test that no error is raised if a CMAPTorsionForce is not present in either end state."""
    (
        cmap_old,
        cmap_new,
        old_num_maps,
        new_num_maps,
        old_num_torsions,
        new_num_torsions
    ) = HybridTopologyFactory._verify_cmap_compatibility(
        None, None
    )
    assert cmap_old is None
    assert cmap_new is None
    assert old_num_maps == 0
    assert new_num_maps == 0
    assert old_num_torsions == 0
    assert new_num_torsions == 0

def test_verify_cmap_missing_cmap_error():
    """Test that an error is raised if a CMAPTorsionForce is only present in one of the end states."""
    with pytest.raises(RuntimeError, match="Inconsistent CMAPTorsionForce between end states expected to be present in both"):
        _ = HybridTopologyFactory._verify_cmap_compatibility(
            None, openmm.CMAPTorsionForce()
        )

def test_verify_cmap_incompatible_maps_error():
    """Test that an error is raised if the number of CMAP terms differ between the end states."""
    old_cmap = openmm.CMAPTorsionForce()
    new_cmap = openmm.CMAPTorsionForce()
    old_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    new_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    new_cmap.addMap(2, [0.0] * 2 * 2)  # add a second map to make them incompatible
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce between end states expected to have same number of maps, found old: 1 and new: 2"):
        _ = HybridTopologyFactory._verify_cmap_compatibility(
            old_cmap, new_cmap
        )

def test_verify_cmap_incompatible_torsions_error():
    """Test that an error is raised if the number of CMAP torsions differ between the end states."""
    old_cmap = openmm.CMAPTorsionForce()
    new_cmap = openmm.CMAPTorsionForce()
    old_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    new_cmap.addMap(2, [0.0] * 2 * 2)  # add one map
    # add torsions
    old_cmap.addTorsion(0, 0, 1, 2, 3, 4, 5, 6, 7)
    new_cmap.addTorsion(0, 0, 1, 2, 3, 4, 5, 6, 7)
    new_cmap.addTorsion(0, 1, 2, 3, 4, 5, 6, 7, 8)  # add a second torsion to make them incompatible
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce between end states expected to have same number of torsions, found old: 1 and new: 2"):
        _ = HybridTopologyFactory._verify_cmap_compatibility(
            old_cmap, new_cmap
        )

def test_cmap_maps_incompatible_error():
    """Test that an error is raised if the CMAP maps differ between the end states using a dummy system.
    In this case the map parameters differ for map index 0 between the old and new systems
    """
    old_system, old_topology, old_positions = _make_system_with_cmap([4])
    new_system, new_topology, new_positions = _make_system_with_cmap([3])
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce map parameters found between end states for map 0 expected"):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so they end up in the environment
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            old_to_new_core_atom_map={}
        )

def test_cmap_torsions_incompatible_error():
    """Test that an error is raised if the CMAP torsions differ between the end states using a dummy system.
    In this case there is an extra cmap torsion in the new system not present in the old system."""
    old_system, old_topology, old_positions = _make_system_with_cmap([4], num_atoms=12)
    new_system, new_topology, new_positions = _make_system_with_cmap([4], num_atoms=12, mapped_torsions=[
        # change the mapped atoms from the default
        (0, 4, 5, 6, 7, 8, 9, 10, 11)
    ])
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce term found between end states for atoms "):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so they end up in the environment
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11},
            old_to_new_core_atom_map={}
        )

def test_cmap_map_index_incompatible_error():
    """Test that an error is raised if the CMAP map indices differ between the end states using a dummy system.
    In this case the map index for the single cmap torsion differs between the old and new systems."""
    old_system, old_topology, old_positions = _make_system_with_cmap([4, 5])
    new_system, new_topology, new_positions = _make_system_with_cmap([4, 5], mapped_torsions=[
        # change the map index from the default
        (1, 0, 1, 2, 3, 4, 5, 6, 7)
    ])
    # modify one of the torsions in the new system to make them incompatible
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce map index found between end states for atoms "):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so they end up in the environment
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            old_to_new_core_atom_map={}
        )

def test_cmap_in_alchemical_region_error():
    """Test that an error is raised if a CMAP torsion is in the alchemical region."""
    old_system, old_topology, old_positions = _make_system_with_cmap([4])
    new_system, new_topology, new_positions = _make_system_with_cmap([4])
    with pytest.raises(RuntimeError, match="Incompatible CMAPTorsionForce term found in alchemical region for old system atoms"):
        _ = HybridTopologyFactory(
            old_system=old_system,
            old_topology=old_topology,
            old_positions=old_positions,
            new_system=new_system,
            new_topology=new_topology,
            new_positions=new_positions,
            # map all atoms so that one of the cmap atoms is in the alchemical core region
            old_to_new_atom_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7},
            old_to_new_core_atom_map={4: 4}  # atom 4 is part of the cmap torsion
        )