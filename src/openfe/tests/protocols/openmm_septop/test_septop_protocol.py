# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import math
import pathlib
from unittest import mock

import gufe
import mdtraj as md
import numpy as np
import openmm
import openmm.app
import openmm.unit
import pytest
from numpy.testing import assert_allclose
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm, to_openmm
from openmm import (
    CustomBondForce,
    CustomCompoundBondForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
)
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from openmmtools.multistate.multistatesampler import MultiStateSampler

import openfe.protocols.openmm_septop
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopComplexAnalysisUnit,
    SepTopComplexRunUnit,
    SepTopComplexSetupUnit,
    SepTopProtocol,
    SepTopProtocolResult,
    SepTopSolventAnalysisUnit,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)
from openfe.protocols.openmm_utils.serialization import deserialize
from openfe.protocols.restraint_utils.geometry.boresch import BoreschRestraintGeometry
from openfe.tests.protocols.conftest import compute_energy
from openfe.tests.protocols.openmm_ahfe.test_ahfe_protocol import (
    _assert_num_forces,
    _verify_alchemical_sterics_force_parameters,
)

from .utils import UNIT_TYPES, _get_units

E_CHARGE = 1.602176634e-19 * openmm.unit.coulomb
EPSILON0 = (
    1e-6
    * 8.8541878128e-12
    / (openmm.unit.AVOGADRO_CONSTANT_NA * E_CHARGE**2)
    * openmm.unit.farad
    / openmm.unit.meter
)
ONE_4PI_EPS0 = 1 / (4 * np.pi * EPSILON0) * EPSILON0.unit * 10.0  # nm -> angstrom


@pytest.fixture()
def default_settings():
    s = SepTopProtocol.default_settings()
    return s


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = SepTopProtocol(
        settings=default_settings,
    )
    assert protocol
    assert protocol.settings == default_settings


def test_serialize_protocol(default_settings):
    protocol = SepTopProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = SepTopProtocol.from_dict(ser)
    assert protocol == ret


def test_repeat_units(benzene_complex_system, toluene_complex_system, default_settings):
    default_settings.protocol_repeats = 3
    protocol = SepTopProtocol(
        settings=default_settings,
    )

    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )

    # 6 protocol unit, 3 per repeat
    pus = list(dag.protocol_units)
    assert len(pus) == 18

    # Check info for each repeat
    for phase in ["solvent", "complex"]:
        setup = _get_units(pus, UNIT_TYPES[phase]["setup"])
        sim = _get_units(pus, UNIT_TYPES[phase]["sim"])
        analysis = _get_units(pus, UNIT_TYPES[phase]["analysis"])

        # Should be 3 of each set
        assert len(setup) == 3
        assert len(sim) == 3
        assert len(analysis) == 3

        # check that the dag chain is correct
        for analysis_pu in analysis:
            repeat_id = analysis_pu.inputs["repeat_id"]
            setup_pu = [
                s for s in setup if (s.inputs["repeat_id"] == repeat_id) and (s.simtype == phase)
            ][0]
            sim_pu = [
                s for s in sim if (s.inputs["repeat_id"] == repeat_id) and (s.simtype == phase)
            ][0]
            assert analysis_pu.inputs["setup"] == setup_pu
            assert analysis_pu.inputs["simulation"] == sim_pu
            assert sim_pu.inputs["setup"] == setup_pu


def test_create_independent_repeat_ids(
    benzene_complex_system, toluene_complex_system, default_settings
):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    # Default protocol is 1 repeat, change to 3 repeats
    default_settings.protocol_repeats = 3
    protocol = SepTopProtocol(
        settings=default_settings,
    )

    dags = []
    for i in range(2):
        dags.append(
            protocol.create(
                stateA=benzene_complex_system,
                stateB=toluene_complex_system,
                mapping=None,
            )
        )

    repeat_ids = set()

    for dag in dags:
        # 3 repeats of 6 units
        assert len(list(dag.protocol_units)) == 18
        for u in dag.protocol_units:
            repeat_ids.add(u.inputs["repeat_id"])

    # one uuid per repeat, so should equal 6
    assert len(repeat_ids) == 6


# Tests for the alchemical systems. This tests were modified from
# femto (https://github.com/Psivant/femto/tree/main)
def compute_interaction_energy(
    epsilon,
    sigma,
    charge,
    distance,
    lambda_vdw: float = 1.0,
    lambda_charges: float = 1.0,
):
    r_electrostatics = distance
    r_vdw = (0.5 * sigma**6 * (1.0 - lambda_vdw) + distance**6) ** (1.0 / 6.0)

    return (
        # vdw
        4.0 * lambda_vdw * epsilon * ((sigma / r_vdw) ** 12 - (sigma / r_vdw) ** 6)
        # electrostatics
        + ONE_4PI_EPS0 * lambda_charges * charge / r_electrostatics
    ) * openmm.unit.kilojoule_per_mole


@pytest.fixture
def three_particle_system():
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    force.setUseDispersionCorrection(False)

    charges = 0.1, 0.2, -0.3
    sigmas = 1.1, 1.2, 1.3
    epsilons = 210, 220, 230

    force.addParticle(charges[0], sigmas[0] * openmm.unit.angstrom, epsilons[0])
    force.addParticle(charges[1], sigmas[1] * openmm.unit.angstrom, epsilons[1])
    force.addParticle(charges[2], sigmas[2] * openmm.unit.angstrom, epsilons[2])

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addForce(force)

    distances = [[0.0, 4.0, 3.0], [4.0, 0.0, 5.0], [3.0, 5.0, 0.0]]

    def interaction_energy_fn(idx_a, idx_b, lambda_vdw: float = 1.0, lambda_charges: float = 1.0):
        epsilon = np.sqrt(epsilons[idx_a] * epsilons[idx_b])
        sigma = 0.5 * (sigmas[idx_a] + sigmas[idx_b])
        charge = charges[idx_a] * charges[idx_b]

        return compute_interaction_energy(
            epsilon, sigma, charge, distances[idx_a][idx_b], lambda_vdw, lambda_charges
        )

    coords = (
        np.array([[0.0, 0.0, 0.0], [distances[0][1], 0.0, 0.0], [0.0, distances[0][2], 0.0]])
        * openmm.unit.angstrom
    )

    return system, coords, interaction_energy_fn


class TestNonbondedInteractions:
    def test_one_ligand(self, three_particle_system):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system

        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(alchemical_atoms=[0], name="A")
        alchemical_system = factory.create_alchemical_system(system, [alchemical_region_A])

        energy_0 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 1.0,
                "lambda_electrostatics_A": 1.0,
            },
        )

        # expect lig_1 + solvent, lig_1 + lig_2 and lig_2 + solvent
        # interaction when
        # lambda=0
        expected_energy_0 = energy_fn(0, 2) + energy_fn(0, 1) + energy_fn(1, 2)
        assert_allclose(energy_0, from_openmm(expected_energy_0), rtol=1e-05)

        # expect only lig_2 + solvent interaction when lambda=1
        energy_1 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 0.0,
                "lambda_electrostatics_A": 0.0,
            },
        )
        expected_energy_1 = energy_fn(1, 2)
        assert_allclose(energy_1, from_openmm(expected_energy_1), rtol=1e-05)

        # expect all particles to interact but only lig - solvent interactions to be
        # scaled
        energy_05 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 0.5,
                "lambda_electrostatics_A": 0.5,
            },
        )
        expected_energy_05 = energy_fn(1, 2) + energy_fn(0, 2, 0.5, 0.5) + energy_fn(0, 1, 0.5, 0.5)
        assert_allclose(energy_05, from_openmm(expected_energy_05), rtol=1e-05)

    def test_two_ligands(self, three_particle_system):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system

        # Do it the openmm way
        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(alchemical_atoms=[0], name="A")
        alchemical_region_B = AlchemicalRegion(alchemical_atoms=[1], name="B")
        alchemical_system = factory.create_alchemical_system(
            system, [alchemical_region_A, alchemical_region_B]
        )
        energy_0 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 1.0,
                "lambda_electrostatics_A": 1.0,
                "lambda_sterics_B": 0.0,
                "lambda_electrostatics_B": 0.0,
            },
        )

        # expect only lig_1 + solvent interaction when lambda=0
        expected_energy_0 = energy_fn(0, 2)
        assert_allclose(energy_0, from_openmm(expected_energy_0), rtol=1e-05)

        # expect only lig_2 + solvent interaction when lambda=1
        energy_1 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 0.0,
                "lambda_electrostatics_A": 0.0,
                "lambda_sterics_B": 1.0,
                "lambda_electrostatics_B": 1.0,
            },
        )
        expected_energy_1 = energy_fn(1, 2)
        assert_allclose(energy_1, from_openmm(expected_energy_1), rtol=1e-05)

        # expect lig_1 + solvent and lig_2 + solvent interaction when lambda=0.5
        # but no lig_1 + lig_2 interaction by default
        energy_05 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 0.5,
                "lambda_electrostatics_A": 0.5,
                "lambda_sterics_B": 0.5,
                "lambda_electrostatics_B": 0.5,
            },
        )
        expected_energy_05 = energy_fn(0, 2, 0.5, 0.5) + energy_fn(1, 2, 0.5, 0.5)
        assert_allclose(energy_05, from_openmm(expected_energy_05), rtol=1e-05)

    def test_two_ligands_charges(self, three_particle_system):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system

        # Do it the openmm way
        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(alchemical_atoms=[0], name="A")
        alchemical_region_B = AlchemicalRegion(alchemical_atoms=[1], name="B")
        alchemical_system = factory.create_alchemical_system(
            system, [alchemical_region_A, alchemical_region_B]
        )
        energy = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                "lambda_sterics_A": 1.0,
                "lambda_electrostatics_A": 0.8,
                "lambda_sterics_B": 1.0,
                "lambda_electrostatics_B": 0.2,
            },
        )
        expected_energy = energy_fn(0, 2, 1.0, 0.8) + energy_fn(1, 2, 1.0, 0.2)

        assert_allclose(energy, from_openmm(expected_energy), rtol=1e-05)


def test_dry_run_benzene_toluene(benzene_toluene_dag, tmp_path):
    prot_units = list(benzene_toluene_dag.protocol_units)

    assert len(prot_units) == 6

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in prot_units if isinstance(u, SepTopSolventRunUnit)]
    complex_setup_unit = [u for u in prot_units if isinstance(u, SepTopComplexSetupUnit)]
    complex_run_unit = [u for u in prot_units if isinstance(u, SepTopComplexRunUnit)]
    assert len(solv_setup_unit) == 1
    assert len(sol_run_unit) == 1
    assert len(complex_setup_unit) == 1
    assert len(complex_run_unit) == 1

    solv_setup_output = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    pdb = md.load_pdb(tmp_path / "topology.pdb")
    assert pdb.n_atoms == 1762
    central_atoms = np.array([[2, 19]], dtype=np.int32)
    distance = md.compute_distances(pdb, central_atoms)[0][0]
    assert np.isclose(distance, 0.8661)
    pdb_file = openmm.app.pdbfile.PDBFile(str(solv_setup_output["topology"]))
    alchem_system = deserialize(solv_setup_output["system"])
    solv_sampler = sol_run_unit[0].run(
        alchem_system,
        pdb_file,
        solv_setup_output["selection_indices"],
        dry=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )["sampler"]  # fmt: skip

    assert solv_sampler.is_periodic
    assert isinstance(solv_sampler, MultiStateSampler)
    assert isinstance(solv_sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)
    assert solv_sampler._thermodynamic_states[1].pressure == 1 * openmm.unit.bar
    # Check we have the right number of atoms in the PDB
    pdb = md.load_pdb(tmp_path / "alchemical_system.pdb")
    assert pdb.n_atoms == 31

    # Test the solvent system
    assert len(alchem_system.getForces()) == 14
    _assert_num_forces(alchem_system, NonbondedForce, 1)
    _assert_num_forces(alchem_system, CustomNonbondedForce, 4)
    _assert_num_forces(alchem_system, CustomBondForce, 4)
    _assert_num_forces(alchem_system, HarmonicBondForce, 2)
    _assert_num_forces(alchem_system, HarmonicAngleForce, 1)
    _assert_num_forces(alchem_system, PeriodicTorsionForce, 1)
    _assert_num_forces(alchem_system, MonteCarloBarostat, 1)

    # Check steric forces
    for f in alchem_system.getForces():
        if isinstance(f, CustomNonbondedForce) and "U_sterics" in f.getEnergyFunction():
            _verify_alchemical_sterics_force_parameters(f)

    complex_setup_output = complex_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    pdb_file = openmm.app.pdbfile.PDBFile(str(complex_setup_output["topology"]))
    alchem_system = deserialize(complex_setup_output["system"])
    complex_sampler = complex_run_unit[0].run(
        alchem_system,
        pdb_file,
        complex_setup_output["selection_indices"],
        dry=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )["sampler"]  # fmt: skip

    assert complex_sampler.is_periodic
    assert isinstance(complex_sampler, MultiStateSampler)
    assert isinstance(complex_sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)
    assert complex_sampler._thermodynamic_states[1].pressure == 1 * openmm.unit.bar
    # Check we have the right number of atoms in the PDB
    pdb = md.load_pdb(tmp_path / "alchemical_system.pdb")
    assert pdb.n_atoms == 2687

    # Test the complex system
    assert len(alchem_system.getForces()) == 15
    _assert_num_forces(alchem_system, NonbondedForce, 1)
    _assert_num_forces(alchem_system, CustomNonbondedForce, 4)
    _assert_num_forces(alchem_system, CustomBondForce, 4)
    _assert_num_forces(alchem_system, HarmonicBondForce, 1)
    _assert_num_forces(alchem_system, HarmonicAngleForce, 1)
    _assert_num_forces(alchem_system, PeriodicTorsionForce, 1)
    _assert_num_forces(alchem_system, CustomCompoundBondForce, 2)
    _assert_num_forces(alchem_system, MonteCarloBarostat, 1)

    # Check steric forces
    for f in alchem_system.getForces():
        if isinstance(f, CustomNonbondedForce) and "U_sterics" in f.getEnergyFunction():
            _verify_alchemical_sterics_force_parameters(f)


@pytest.mark.parametrize("method", ["repex", "sams", "independent"])
def test_dry_run_methods(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
    method,
):
    protocol_dry_settings.solvent_simulation_settings.sampler_method = method
    protocol_dry_settings.complex_simulation_settings.sampler_method = method
    protocol_dry_settings.complex_output_settings.output_indices = "resname UNK"
    protocol_dry_settings.solvent_output_settings.output_indices = "resname UNK"

    protocol = SepTopProtocol(
        settings=protocol_dry_settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag_units = list(dag.protocol_units)
    # Only check the cutoff for the Solvent SetUp Unit
    solv_setup_unit = [u for u in dag_units if isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in dag_units if isinstance(u, SepTopSolventRunUnit)]

    solv_setup_output = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    pdb_file = openmm.app.pdbfile.PDBFile(str(solv_setup_output["topology"]))
    alchem_system = deserialize(solv_setup_output["system"])
    solv_sampler = sol_run_unit[0].run(
        alchem_system,
        pdb_file,
        solv_setup_output["selection_indices"],
        dry=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )["sampler"]  # fmt: skip

    assert isinstance(solv_sampler, MultiStateSampler)
    assert solv_sampler.is_periodic
    assert isinstance(solv_sampler._thermodynamic_states[0].barostat, MonteCarloBarostat)
    assert solv_sampler._thermodynamic_states[1].pressure == 1 * openmm.unit.bar

    # Check we have the right number of atoms in the PDB
    pdb = md.load_pdb(tmp_path / "alchemical_system.pdb")
    assert pdb.n_atoms == 27


@pytest.mark.parametrize(
    "pressure",
    [
        1.0,
        0.9,
        1.1,
    ],
)
def test_dry_run_ligand_system_pressure(
    pressure,
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
):
    """
    Test that the right nonbonded cutoff is propagated to the system.
    """
    # openfe settings requires openff/pint units
    protocol_dry_settings.thermo_settings.pressure = pressure * offunit.bar

    protocol = SepTopProtocol(
        settings=protocol_dry_settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag_units = list(dag.protocol_units)
    # Only check the cutoff for the Solvent SetUp Unit
    solv_setup_unit = [u for u in dag_units if isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in dag_units if isinstance(u, SepTopSolventRunUnit)]

    solv_setup_output = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    pdb_file = openmm.app.pdbfile.PDBFile(str(solv_setup_output["topology"]))
    alchem_system = deserialize(solv_setup_output["system"])
    solv_sampler = sol_run_unit[0].run(
        alchem_system,
        pdb_file,
        solv_setup_output["selection_indices"],
        dry=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )["sampler"]  # fmt: skip

    # at this point, the units will be in openmm units
    assert solv_sampler._thermodynamic_states[1].pressure == pressure * openmm.unit.bar


def test_virtual_sites_no_reassign(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
):
    """
    Test that an error is raised when not reassigning velocities
    in a system with virtual site.
    """
    protocol_dry_settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",
        "amber/tip4pew_standard.xml",  # FF with VS
    ]
    protocol_dry_settings.solvent_solvation_settings.solvent_model = "tip4pew"
    protocol_dry_settings.solvent_integrator_settings.reassign_velocities = False

    protocol = SepTopProtocol(
        settings=protocol_dry_settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag_units = list(dag.protocol_units)
    # Only check the Solvent Unit
    solv_setup_unit = [u for u in dag_units if isinstance(u, SepTopSolventSetupUnit)]
    solv_run_unit = [u for u in dag_units if isinstance(u, SepTopSolventRunUnit)]

    setup_results = solv_setup_unit[0].run(
        dry=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )

    pdb_file = openmm.app.pdbfile.PDBFile(str(setup_results["topology"]))

    with pytest.raises(ValueError, match="are unstable"):
        _ = solv_run_unit[0].run(
            setup_results["alchem_system"],
            pdb_file,
            setup_results["selection_indices"],
            dry=True,
            scratch_basepath=tmp_path,
            shared_basepath=tmp_path,
        )  # fmt: skip


@pytest.mark.parametrize(
    "cutoff",
    [1.0 * offunit.nanometer, 12.0 * offunit.angstrom, 0.9 * offunit.nanometer],
)
def test_dry_run_ligand_system_cutoff(
    cutoff,
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
):
    """
    Test that the right nonbonded cutoff is propagated to the system.
    """
    protocol_dry_settings.solvent_solvation_settings.solvent_padding = 1.9 * offunit.nanometer
    protocol_dry_settings.forcefield_settings.nonbonded_cutoff = cutoff

    protocol = SepTopProtocol(
        settings=protocol_dry_settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag_units = list(dag.protocol_units)
    # Only check the cutoff for the Solvent SetUp Unit
    solv_setup_unit = [u for u in dag_units if isinstance(u, SepTopSolventSetupUnit)]

    serialized_system = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )["system"]
    system = deserialize(serialized_system)
    nbfs = [
        f
        for f in system.getForces()
        if isinstance(f, CustomNonbondedForce) or isinstance(f, NonbondedForce)
    ]

    for f in nbfs:
        f_cutoff = from_openmm(f.getCutoffDistance())
        assert f_cutoff == cutoff


def test_dry_run_benzene_toluene_tip4p(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
):
    protocol_dry_settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    protocol_dry_settings.solvent_solvation_settings.solvent_model = "tip4pew"
    protocol_dry_settings.solvent_integrator_settings.reassign_velocities = True

    protocol = SepTopProtocol(settings=protocol_dry_settings)

    # Create DAG from protocol, get the solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )

    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 6

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in prot_units if isinstance(u, SepTopSolventRunUnit)]

    assert len(solv_setup_unit) == 1
    assert len(sol_run_unit) == 1

    solv_setup_output = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    pdb_file = openmm.app.pdbfile.PDBFile(str(solv_setup_output["topology"]))
    alchem_system = deserialize(solv_setup_output["system"])
    solv_run = sol_run_unit[0].run(
        alchem_system,
        pdb_file,
        solv_setup_output["selection_indices"],
        dry=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )["sampler"]  # fmt: skip

    assert solv_run.is_periodic


def test_dry_run_benzene_toluene_noncubic(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
):
    protocol_dry_settings.solvent_solvation_settings.solvent_padding = 1.5 * offunit.nanometer
    protocol_dry_settings.solvent_solvation_settings.box_shape = "dodecahedron"

    protocol = SepTopProtocol(settings=protocol_dry_settings)

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )

    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 6

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]

    assert len(solv_setup_unit) == 1

    solv_setup_output = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    serialized_system = solv_setup_output["system"]
    system = deserialize(serialized_system)
    vectors = system.getDefaultPeriodicBoxVectors()
    width = float(from_openmm(vectors)[0][0].to("nanometer").m)

    # dodecahedron has the following shape:
    # [width, 0, 0], [0, width, 0], [0.5, 0.5, 0.5 * sqrt(2)] * width

    expected_vectors = [
        [width, 0, 0],
        [0, width, 0],
        [0.5 * width, 0.5 * width, 0.5 * math.sqrt(2) * width],
    ] * offunit.nanometer
    assert_allclose(
        expected_vectors,
        from_openmm(vectors),
    )


def test_dry_run_solv_user_charges_benzene_toluene(
    benzene_modifications,
    T4_protein_component,
    tmp_path,
    protocol_dry_settings,
):
    """
    Create a test system with fictitious user supplied charges and
    ensure that they are properly passed through to the constructed
    alchemical system.
    """
    protocol = SepTopProtocol(settings=protocol_dry_settings)

    def assign_fictitious_charges(offmol):
        """
        Get a random array of fake partial charges for your offmol.
        """
        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
        rand_arr[-1] = -sum(rand_arr[:-1])
        return rand_arr * offunit.elementary_charge

    def check_partial_charges(offmol):
        offmol_pchgs = assign_fictitious_charges(offmol)
        offmol.partial_charges = offmol_pchgs
        smc = openfe.SmallMoleculeComponent.from_openff(offmol)

        # check propchgs
        prop_chgs = smc.to_dict()["molprops"]["atom.dprop.PartialCharge"]
        prop_chgs = np.array(prop_chgs.split(), dtype=float)
        np.testing.assert_allclose(prop_chgs, offmol_pchgs)
        return smc, prop_chgs

    benzene_offmol = benzene_modifications["benzene"].to_openff()
    toluene_offmol = benzene_modifications["toluene"].to_openff()

    benzene_smc, benzene_charge = check_partial_charges(benzene_offmol)
    toluene_smc, toluene_charge = check_partial_charges(toluene_offmol)

    # Create ChemicalSystems
    stateA = ChemicalSystem(
        {
            "benzene": benzene_smc,
            "T4l": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "toluene": toluene_smc,
            "T4l": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]
    complex_setup_unit = [u for u in prot_units if isinstance(u, SepTopComplexSetupUnit)]

    # check sol_unit charges
    serialized_system = solv_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )["system"]
    system = deserialize(serialized_system)
    nonbond = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    assert len(nonbond) == 1

    # loop through the 12 benzene atoms
    # partial charge is stored in the offset
    for i in range(12):
        offsets = nonbond[0].getParticleParameterOffset(i)
        c = ensure_quantity(offsets[2], "openff")
        assert pytest.approx(c) == benzene_charge[i]
    # loop through 15 toluene atoms
    for inx, i in enumerate(range(12, 27)):
        offsets = nonbond[0].getParticleParameterOffset(i)
        c = ensure_quantity(offsets[2], "openff")
        assert pytest.approx(c) == toluene_charge[inx]

    # check complex_unit charges
    serialized_system = complex_setup_unit[0].run(
        dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )["system"]
    system = deserialize(serialized_system)
    nonbond = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    assert len(nonbond) == 1

    # loop through the 12 benzene atoms
    # partial charge is stored in the offset
    for i in range(12):
        offsets = nonbond[0].getParticleParameterOffset(i)
        c = ensure_quantity(offsets[2], "openff")
        assert pytest.approx(c) == benzene_charge[i]
    # loop through 15 toluene atoms
    for inx, i in enumerate(range(12, 27)):
        offsets = nonbond[0].getParticleParameterOffset(i)
        c = ensure_quantity(offsets[2], "openff")
        assert pytest.approx(c) == toluene_charge[inx]


def test_high_timestep(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path,
    protocol_dry_settings,
):
    protocol_dry_settings.forcefield_settings.hydrogen_mass = 1.0
    protocol_dry_settings.forcefield_settings.hydrogen_mass = 1.0

    protocol = SepTopProtocol(settings=protocol_dry_settings)

    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    errmsg = "too large for hydrogen mass"
    with pytest.raises(ValueError, match=errmsg):
        prot_units[0].run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)


def test_bad_sampler():
    class FakeSimSettings(gufe.settings.SettingsBaseModel):
        sampler_method: str = "foo bar"

    errmsg = "Unknown sampler foo bar"
    with pytest.raises(AttributeError, match=errmsg):
        SepTopSolventRunUnit._get_sampler(
            integrator=None,
            reporter=None,
            simulation_settings=FakeSimSettings(),
            thermodynamic_settings=None,
            compound_states=None,
            sampler_states=None,
            platform=None,
            restart=False,
        )


@pytest.fixture
def T4L_xml(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path_factory,
    protocol_dry_settings,
):
    # Fixing the number of solvent molecules in the solvent settings
    # to test against reference xml
    protocol_dry_settings.solvent_solvation_settings.solvent_padding = None
    protocol_dry_settings.solvent_solvation_settings.number_of_solvent_molecules = 364
    protocol_dry_settings.forcefield_settings.small_molecule_forcefield = "openff-2.1.1"
    protocol = SepTopProtocol(settings=protocol_dry_settings)

    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    # Get the SepTopSolventSetupUnit
    prot_units = list(dag.protocol_units)
    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]

    tmp = tmp_path_factory.mktemp("xml_reg")

    dryrun = solv_setup_unit[0].run(dry=True, shared_basepath=tmp)

    system = dryrun["system"]
    return deserialize(system)


class TestT4LXmlRegression:
    """Generates SepTop system XML (solvent) and performs regression test"""

    @staticmethod
    def test_particles(T4L_xml, T4L_septop_reference_xml):
        nr_particles = T4L_xml.getNumParticles()
        nr_particles_ref = T4L_septop_reference_xml.getNumParticles()
        assert nr_particles == nr_particles_ref
        particle_masses = [T4L_xml.getParticleMass(i) for i in range(nr_particles)]
        particle_masses_ref = [
            T4L_septop_reference_xml.getParticleMass(i) for i in range(nr_particles)
        ]
        assert particle_masses

        for a, b in zip(particle_masses, particle_masses_ref):
            assert a == b

    @staticmethod
    def test_constraints(T4L_xml, T4L_septop_reference_xml):
        nr_constraints = T4L_xml.getNumConstraints()
        nr_constraints_ref = T4L_septop_reference_xml.getNumConstraints()
        assert nr_constraints == nr_constraints_ref
        constraints = [T4L_xml.getConstraintParameters(i) for i in range(nr_constraints)]
        constraints_ref = [
            T4L_septop_reference_xml.getConstraintParameters(i) for i in range(nr_constraints)
        ]
        assert constraints

        for a, b in zip(constraints, constraints_ref):
            # Particle 1
            assert a[0] == b[0]
            # Particle 2
            assert a[1] == b[1]
            # Constraint Quantity
            assert a[2] == b[2]


@pytest.mark.slow
class TestA2AMembraneDryRun:
    solvent = SolventComponent(ion_concentration=0 * offunit.molar)
    num_all_not_water = 16116
    num_complex_atoms = 39462
    num_ligand_atoms_A = 36
    num_ligand_atoms_B = 36

    @pytest.fixture(scope="class")
    def settings(self):
        s = SepTopProtocol.default_settings()
        s.protocol_repeats = 1
        s.engine_settings.compute_platform = "cpu"
        s.complex_output_settings.output_indices = "not water"
        s.complex_solvation_settings.box_shape = "dodecahedron"
        s.complex_solvation_settings.solvent_padding = 0.9 * offunit.nanometer
        s.solvent_solvation_settings.box_shape = "cube"
        return s

    @pytest.fixture(scope="function")
    def dag(self, settings, a2a_ligands, a2a_protein_membrane_component):
        stateA = ChemicalSystem(
            {
                "ligandA": a2a_ligands[0],
                "protein": a2a_protein_membrane_component,
                "solvent": self.solvent,
            }
        )

        stateB = ChemicalSystem(
            {
                "ligandB": a2a_ligands[1],
                "protein": a2a_protein_membrane_component,
                "solvent": self.solvent,
            }
        )

        # adaptive settings
        protocol_settings = SepTopProtocol._adaptive_settings(
            stateA=stateA,
            stateB=stateB,
            initial_settings=settings,
        )
        protocol = SepTopProtocol(settings=protocol_settings)

        return protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )

    @pytest.fixture(scope="function")
    def complex_setup_units(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, SepTopComplexSetupUnit)]

    @pytest.fixture(scope="function")
    def complex_run_units(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, SepTopComplexRunUnit)]

    @pytest.fixture(scope="function")
    def complex_analysis_unit(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, SepTopComplexAnalysisUnit)]

    @pytest.fixture(scope="function")
    def solvent_setup_units(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, SepTopSolventSetupUnit)]

    @pytest.fixture(scope="function")
    def solvent_run_units(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, SepTopSolventRunUnit)]

    @pytest.fixture(scope="function")
    def solvent_analysis_unit(self, dag):
        return [u for u in dag.protocol_units if isinstance(u, SepTopSolventAnalysisUnit)]

    def test_number_of_units(
        self, dag, complex_setup_units, complex_run_units, solvent_setup_units, solvent_run_units
    ):
        assert len(list(dag.protocol_units)) == 6
        assert len(complex_setup_units) == 1
        assert len(complex_run_units) == 1
        assert len(solvent_setup_units) == 1
        assert len(solvent_run_units) == 1

    def _assert_force_num(self, system, forcetype, number):
        forces = [f for f in system.getForces() if isinstance(f, forcetype)]
        assert len(forces) == number

    def _assert_expected_alchemical_forces(self, system, complexed: bool, settings):
        """
        Assert the forces expected in the alchemical system.
        """
        if complexed:
            barostat_type = MonteCarloMembraneBarostat
            self._assert_force_num(system, HarmonicBondForce, 1)
            # Two custom bonds for the two Boresch restraints
            self._assert_force_num(system, CustomCompoundBondForce, 2)
            assert len(system.getForces()) == 15
        else:
            # Extra bond in the solvent
            self._assert_force_num(system, HarmonicBondForce, 2)
            assert len(system.getForces()) == 14
            barostat_type = MonteCarloBarostat

        self._assert_force_num(system, NonbondedForce, 1)
        self._assert_force_num(system, CustomNonbondedForce, 4)
        self._assert_force_num(system, CustomBondForce, 4)
        self._assert_force_num(system, HarmonicAngleForce, 1)
        self._assert_force_num(system, PeriodicTorsionForce, 1)
        self._assert_force_num(system, barostat_type, 1)

        # Check the nonbonded force has the right contents
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert len(nonbond) == 1
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.PME
        assert (
            from_openmm(nonbond[0].getCutoffDistance())
            == settings.forcefield_settings.nonbonded_cutoff
        )

        # Check the barostat made it all the way through
        barostat = [f for f in system.getForces() if isinstance(f, barostat_type)]
        assert len(barostat) == 1
        assert barostat[0].getFrequency() == int(
            settings.complex_integrator_settings.barostat_frequency.m
        )
        assert barostat[0].getDefaultPressure() == to_openmm(settings.thermo_settings.pressure)
        assert barostat[0].getDefaultTemperature() == to_openmm(
            settings.thermo_settings.temperature
        )

    def _assert_expected_nonalchemical_forces(self, system, complexed: bool, settings):
        """
        Assert the forces expected in the non-alchemical system.
        """
        if complexed:
            barostat_type = MonteCarloMembraneBarostat
        else:
            barostat_type = MonteCarloBarostat
        self._assert_force_num(system, NonbondedForce, 1)
        self._assert_force_num(system, HarmonicBondForce, 1)
        self._assert_force_num(system, HarmonicAngleForce, 1)
        self._assert_force_num(system, PeriodicTorsionForce, 1)
        self._assert_force_num(system, barostat_type, 1)

        assert len(system.getForces()) == 5

        # Check that the nonbonded force has the right contents
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
        assert len(nonbond) == 1
        assert nonbond[0].getNonbondedMethod() == NonbondedForce.PME
        assert (
            from_openmm(nonbond[0].getCutoffDistance())
            == settings.forcefield_settings.nonbonded_cutoff
        )

        # Check the barostat made it all the way through
        barostat = [f for f in system.getForces() if isinstance(f, barostat_type)]
        assert len(barostat) == 1
        assert barostat[0].getFrequency() == int(
            settings.complex_integrator_settings.barostat_frequency.m
        )
        assert barostat[0].getDefaultPressure() == to_openmm(settings.thermo_settings.pressure)
        assert barostat[0].getDefaultTemperature() == to_openmm(
            settings.thermo_settings.temperature
        )

    def _verify_sampler(self, sampler, complexed: bool, settings):
        """
        Utility to verify the contents of the sampler.
        """
        assert sampler.is_periodic
        assert isinstance(sampler, MultiStateSampler)
        if complexed:
            barostat_type = MonteCarloMembraneBarostat
        else:
            barostat_type = MonteCarloBarostat
        assert isinstance(sampler._thermodynamic_states[0].barostat, barostat_type)
        assert sampler._thermodynamic_states[1].pressure == to_openmm(
            settings.thermo_settings.pressure
        )
        for state in sampler._thermodynamic_states:
            system = state.get_system(remove_thermostat=True)
            self._assert_expected_alchemical_forces(system, complexed, settings)

    @staticmethod
    def _test_orthogonal_vectors(system):
        """Test that the system has an orthorhombic (rectangular) periodic box."""
        vectors = system.getDefaultPeriodicBoxVectors()
        vectors = from_openmm(vectors)  # convert to a Quantity array

        # Extract box lengths in nanometers
        width_x, width_y, width_z = [v[i].to("nanometer").m for i, v in enumerate(vectors)]

        # Expected orthogonal box (axis-aligned)
        expected_vectors = (
            np.array(
                [
                    [width_x, 0, 0],
                    [0, width_y, 0],
                    [0, 0, width_z],
                ]
            )
            * offunit.nanometer
        )

        assert_allclose(
            vectors, expected_vectors, atol=1e-5, err_msg=f"Box is not orthogonal:\n{vectors}"
        )

    @staticmethod
    def _test_cubic_vectors(system):
        # cube is an identity matrix
        vectors = system.getDefaultPeriodicBoxVectors()
        width = float(from_openmm(vectors)[0][0].to("nanometer").m)

        expected_vectors = [
            [width, 0, 0],
            [0, width, 0],
            [0, 0, width],
        ] * offunit.nanometer

        assert_allclose(
            expected_vectors,
            from_openmm(vectors),
        )

    def test_complex_dry_run(self, complex_setup_units, complex_run_units, tmpdir):
        with tmpdir.as_cwd():
            # Get adaptive settings
            adaptive_settings = complex_setup_units[0]._inputs["protocol"].settings
            # Check that adaptive settings changed the barostat to membrane barostat
            assert (
                adaptive_settings.complex_integrator_settings.barostat
                == "MonteCarloMembraneBarostat"
            )
            complex_setup_output = complex_setup_units[0].run(dry=True)
            pdb_file = openmm.app.pdbfile.PDBFile(str(complex_setup_output["topology"]))
            system = deserialize(complex_setup_output["system"])
            indices = complex_setup_output["selection_indices"]
            data = complex_run_units[0].run(system, pdb_file, indices, dry=True)  # fmt: skip
            # Check the sampler
            self._verify_sampler(data["sampler"], complexed=True, settings=adaptive_settings)

            # Check the alchemical system
            self._assert_expected_alchemical_forces(
                complex_setup_output["alchem_restrained_system"],
                complexed=True,
                settings=adaptive_settings,
            )
            self._test_orthogonal_vectors(complex_setup_output["alchem_restrained_system"])

            # Check the non-alchemical system
            self._assert_expected_nonalchemical_forces(
                complex_setup_output["system_AB"], complexed=True, settings=adaptive_settings
            )
            self._test_orthogonal_vectors(complex_setup_output["system_AB"])
            # Check the box vectors haven't changed (they shouldn't have because we didn't do MD)
            assert_allclose(
                from_openmm(
                    complex_setup_output["alchem_restrained_system"].getDefaultPeriodicBoxVectors()
                ),
                from_openmm(complex_setup_output["system_AB"].getDefaultPeriodicBoxVectors()),
            )

            # Check the PDB
            pdb = md.load_pdb("alchemical_system.pdb")
            assert pdb.n_atoms == self.num_all_not_water

            full_pdb = md.load_pdb("topology.pdb")
            assert full_pdb.n_atoms == self.num_complex_atoms

    def test_solvent_dry_run(self, solvent_setup_units, solvent_run_units, settings, tmpdir):
        with tmpdir.as_cwd():
            solv_setup_output = solvent_setup_units[0].run(dry=True)
            pdb_file = openmm.app.pdbfile.PDBFile(str(solv_setup_output["topology"]))
            system = deserialize(solv_setup_output["system"])
            indices = solv_setup_output["selection_indices"]
            data = solvent_run_units[0].run(system, pdb_file, indices, dry=True)  # fmt: skip

            # Check the sampler
            self._verify_sampler(data["sampler"], complexed=False, settings=settings)

            # Check the alchemical system
            self._assert_expected_alchemical_forces(
                solv_setup_output["alchem_restrained_system"], complexed=False, settings=settings
            )
            self._test_cubic_vectors(solv_setup_output["alchem_restrained_system"])

            # Check the alchemical indices
            expected_indices = [i for i in range(self.num_ligand_atoms_A + self.num_ligand_atoms_B)]
            assert expected_indices == solv_setup_output["selection_indices"].tolist()

            # Check the non-alchemical system
            self._assert_expected_nonalchemical_forces(
                solv_setup_output["system_AB"], complexed=False, settings=settings
            )
            self._test_cubic_vectors(solv_setup_output["system_AB"])

            # Check the PDB
            pdb = md.load_pdb("alchemical_system.pdb")
            assert pdb.n_atoms == (self.num_ligand_atoms_A + self.num_ligand_atoms_B)
