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
import openfe.protocols.openmm_septop
import openmm
import openmm.app
import openmm.unit
import pytest
from numpy.testing import assert_allclose
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import (
    SepTopComplexRunUnit,
    SepTopComplexSetupUnit,
    SepTopProtocol,
    SepTopProtocolResult,
    SepTopSolventRunUnit,
    SepTopSolventSetupUnit,
)
from openfe.protocols.openmm_septop.equil_septop_method import (
    _check_alchemical_charge_difference,
)
from openfe.protocols.openmm_septop.utils import deserialize
from openfe.protocols.openmm_utils import system_validation
from openfe.tests.protocols.conftest import compute_energy
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm
from openmm import CustomNonbondedForce, MonteCarloBarostat, NonbondedForce
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from openmmtools.multistate.multistatesampler import MultiStateSampler

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
    return SepTopProtocol.default_settings()


def test_create_default_settings():
    settings = SepTopProtocol.default_settings()
    assert settings


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [0.0, -1], "vdw": [0.0, 1.0], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1], "vdw": [0.0, 1.5], "restraints": [0.0, 1.0]},
        {"elec": [0.0, 1], "vdw": [0.0, 1], "restraints": [-0.1, 1.0]},
    ],
)
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.complex_lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_A = val["elec"]
        lambda_settings.lambda_vdw_A = val["vdw"]
        lambda_settings.lambda_restraints_A = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {
            "elec": [0.0, 0.1, 0.0],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 1.0],
        },
        {
            "elec": [0.0, 0.0, 0.0],
            "vdw": [0.0, 1.0, 0.0],
            "restraints": [0.0, 1.0, 1.0],
        },
        {
            "elec": [0.0, 0.0, 0.0],
            "vdw": [0.0, 1.0, 1.0],
            "restraints": [0.0, 1.0, 0.0],
        },
    ],
)
def test_monotonic_lambda_windows(val, default_settings):
    errmsg = "The lambda schedule is not monotonic."
    lambda_settings = default_settings.complex_lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_A = val["elec"]
        lambda_settings.lambda_vdw_A = val["vdw"]
        lambda_settings.lambda_restraints_A = val["restraints"]


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec_A = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw_A = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints_A = val["restraints"]
    default_settings.complex_lambda_settings.lambda_elec_B = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw_B = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints_B = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        f"Number of replicas {n_replicas} does not equal the"
        f" number of lambda windows {len(val['vdw'])}"
    )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 1.0, 1.0], "vdw": [0.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nwindows(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec_A = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw_A = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints_A = val["restraints"]
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount of lambda "
        "windows. Got 3 and 19 elec lambda windows"
    )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize(
    "val",
    [
        {"elec": [1.0, 0.5], "vdw": [1.0, 1.0], "restraints": [0.0, 0.0]},
    ],
)
def test_validate_lambda_schedule_nakedcharge(val, default_settings):
    default_settings.complex_lambda_settings.lambda_elec_A = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw_A = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints_A = val["restraints"]
    default_settings.complex_lambda_settings.lambda_elec_B = val["elec"]
    default_settings.complex_lambda_settings.lambda_vdw_B = val["vdw"]
    default_settings.complex_lambda_settings.lambda_restraints_B = val["restraints"]
    n_replicas = 2
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    default_settings.solvent_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        "interactions: State A: l"
    )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.complex_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.complex_lambda_settings,
            default_settings.solvent_simulation_settings,
        )


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

    dag1 = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag2 = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    # print([u for u in dag1.protocol_units])
    repeat_ids = set()
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])

    # There are 4 units per repeat per DAG: 4 * 3 * 2 = 24
    assert len(repeat_ids) == 24


def test_check_alchem_charge_diff(charged_benzene_modifications):
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        _check_alchemical_charge_difference(
            charged_benzene_modifications["benzene"],
            charged_benzene_modifications["benzoic_acid"],
        )


def test_charge_error_create(
    charged_benzene_modifications, T4_protein_component, default_settings
):
    protocol = SepTopProtocol(
        settings=default_settings,
    )
    stateA = ChemicalSystem(
        {
            "benzene": charged_benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "benzoic": charged_benzene_modifications["benzoic_acid"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )


@pytest.mark.parametrize(
    "fail_endstate, system_A, system_B",
    [
        ("stateA", "benzene_system", "benzene_complex_system"),
        ("stateB", "benzene_complex_system", "benzene_system"),
    ],
)
def test_validate_complex_endstates_protcomp(
    request, system_A, system_B, fail_endstate
):

    with pytest.raises(
        ValueError, match=f"No ProteinComponent found in {fail_endstate}"
    ):
        SepTopProtocol._validate_complex_endstates(
            request.getfixturevalue(system_A),
            request.getfixturevalue(system_B),
        )


@pytest.fixture
def T4L_benzene_vacuum(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
        }
    )


@pytest.mark.parametrize(
    "fail_endstate, system_A, system_B",
    [
        ("stateA", "T4L_benzene_vacuum", "benzene_complex_system"),
        ("stateB", "benzene_complex_system", "T4L_benzene_vacuum"),
    ],
)
def test_validate_complex_endstates_nosolvcomp(
    request,
    system_A,
    system_B,
    fail_endstate,
):

    with pytest.raises(
        ValueError, match=f"No SolventComponent found in {fail_endstate}"
    ):
        SepTopProtocol._validate_complex_endstates(
            request.getfixturevalue(system_A),
            request.getfixturevalue(system_B),
        )


@pytest.fixture
def T4L_system(T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "solvent": openfe.SolventComponent(),
            "protein": T4_protein_component,
        }
    )


@pytest.mark.parametrize(
    "fail_endstate, system_A, system_B",
    [
        ("stateA", "T4L_system", "benzene_complex_system"),
        ("stateB", "benzene_complex_system", "T4L_system"),
    ],
)
def test_validate_alchem_comps_missing(
    request,
    system_A,
    system_B,
    fail_endstate,
):

    alchem_comps = system_validation.get_alchemical_components(
        request.getfixturevalue(system_A),
        request.getfixturevalue(system_B),
    )

    with pytest.raises(
        ValueError,
        match="one alchemical component must be " f"present in {fail_endstate}.",
    ):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_comps_toomanyA(
    benzene_modifications,
    T4_protein_component,
):
    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "toluene": benzene_modifications["toluene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "phenol": benzene_modifications["phenol"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps["stateA"]) == 2

    assert len(alchem_comps["stateB"]) == 1

    with pytest.raises(
        ValueError, match="present in stateA. Found 2 alchemical components."
    ):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_nonsmc(
    benzene_modifications,
    T4_protein_component,
):
    stateA = ChemicalSystem(
        {"benzene": benzene_modifications["benzene"], "solvent": SolventComponent()}
    )

    stateB = ChemicalSystem(
        {"benzene": benzene_modifications["benzene"], "protein": T4_protein_component}
    )

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match="Only SmallMoleculeComponent alchemical"):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


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

    def interaction_energy_fn(
        idx_a, idx_b, lambda_vdw: float = 1.0, lambda_charges: float = 1.0
    ):
        epsilon = np.sqrt(epsilons[idx_a] * epsilons[idx_b])
        sigma = 0.5 * (sigmas[idx_a] + sigmas[idx_b])
        charge = charges[idx_a] * charges[idx_b]

        return compute_interaction_energy(
            epsilon, sigma, charge, distances[idx_a][idx_b], lambda_vdw, lambda_charges
        )

    coords = (
        np.array(
            [[0.0, 0.0, 0.0], [distances[0][1], 0.0, 0.0], [0.0, distances[0][2], 0.0]]
        )
        * openmm.unit.angstrom
    )

    return system, coords, interaction_energy_fn


class TestNonbondedInteractions:
    def test_one_ligand(self, three_particle_system):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system

        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(alchemical_atoms=[0], name="A")
        alchemical_system = factory.create_alchemical_system(
            system, [alchemical_region_A]
        )

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
        expected_energy_05 = (
            energy_fn(1, 2) + energy_fn(0, 2, 0.5, 0.5) + energy_fn(0, 1, 0.5, 0.5)
        )
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


@pytest.fixture
def benzene_toluene_dag(
    benzene_complex_system,
    toluene_complex_system,
    default_settings,
):
    default_settings.protocol_repeats = 1
    protocol = SepTopProtocol(settings=default_settings)

    return protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )


def test_dry_run_benzene_toluene(benzene_toluene_dag, tmpdir):

    prot_units = list(benzene_toluene_dag.protocol_units)

    assert len(prot_units) == 4

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in prot_units if isinstance(u, SepTopSolventRunUnit)]
    complex_setup_unit = [
        u for u in prot_units if isinstance(u, SepTopComplexSetupUnit)
    ]
    complex_run_unit = [u for u in prot_units if isinstance(u, SepTopComplexRunUnit)]
    assert len(solv_setup_unit) == 1
    assert len(sol_run_unit) == 1
    assert len(complex_setup_unit) == 1
    assert len(complex_run_unit) == 1

    with tmpdir.as_cwd():
        solv_setup_output = solv_setup_unit[0].run(dry=True)
        pdb = md.load_pdb("topology.pdb")
        assert pdb.n_atoms == 1346
        central_atoms = np.array([[2, 19]], dtype=np.int32)
        distance = md.compute_distances(pdb, central_atoms)[0][0]
        assert np.isclose(distance, 0.8661)
        serialized_topology = solv_setup_output["topology"]
        serialized_system = solv_setup_output["system"]
        solv_sampler = sol_run_unit[0].run(
            serialized_system, serialized_topology, dry=True
        )["debug"]["sampler"]
        assert solv_sampler.is_periodic
        assert isinstance(solv_sampler, MultiStateSampler)
        assert isinstance(
            solv_sampler._thermodynamic_states[0].barostat, MonteCarloBarostat
        )
        assert solv_sampler._thermodynamic_states[1].pressure == 1 * openmm.unit.bar
        # Check we have the right number of atoms in the PDB
        pdb = md.load_pdb("alchemical_system.pdb")
        assert pdb.n_atoms == 29

        complex_setup_output = complex_setup_unit[0].run(dry=True)
        serialized_topology = complex_setup_output["topology"]
        serialized_system = complex_setup_output["system"]
        complex_sampler = complex_run_unit[0].run(
            serialized_system, serialized_topology, dry=True
        )["debug"]["sampler"]
        assert complex_sampler.is_periodic
        assert isinstance(complex_sampler, MultiStateSampler)
        assert isinstance(
            complex_sampler._thermodynamic_states[0].barostat, MonteCarloBarostat
        )
        assert complex_sampler._thermodynamic_states[1].pressure == 1 * openmm.unit.bar
        # Check we have the right number of atoms in the PDB
        pdb = md.load_pdb("alchemical_system.pdb")
        assert pdb.n_atoms == 2713


@pytest.mark.parametrize('method', ['repex', 'sams', 'independent'])
def test_dry_run_methods(
    benzene_complex_system,
    toluene_complex_system,
    tmpdir,
    default_settings,
    method,
):

    default_settings.solvent_simulation_settings.sampler_method = method
    default_settings.complex_simulation_settings.sampler_method = method
    default_settings.protocol_repeats = 1
    default_settings.complex_output_settings.output_indices = 'resname UNK'
    default_settings.solvent_output_settings.output_indices = 'resname UNK'

    protocol = SepTopProtocol(
        settings=default_settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag_units = list(dag.protocol_units)
    # Only check the cutoff for the Solvent SetUp Unit
    solv_setup_unit = [u for u in dag_units if
                       isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in dag_units if
                    isinstance(u, SepTopSolventRunUnit)]
    with tmpdir.as_cwd():
        solv_setup_output = solv_setup_unit[0].run(dry=True)
        serialized_topology = solv_setup_output["topology"]
        serialized_system = solv_setup_output["system"]
        solv_sampler = sol_run_unit[0].run(
            serialized_system, serialized_topology, dry=True
        )["debug"]["sampler"]

        assert isinstance(solv_sampler, MultiStateSampler)
        assert solv_sampler.is_periodic
        assert isinstance(solv_sampler._thermodynamic_states[0].barostat,
                          MonteCarloBarostat)
        assert solv_sampler._thermodynamic_states[1].pressure == 1 * openmm.unit.bar

        # Check we have the right number of atoms in the PDB
        pdb = md.load_pdb('alchemical_system.pdb')
        assert pdb.n_atoms == 27


@pytest.mark.parametrize(
    "pressure",
    [
        1.0 * openmm.unit.atmosphere,
        0.9 * openmm.unit.atmosphere,
        1.1 * openmm.unit.atmosphere,
    ],
)
def test_dry_run_ligand_system_pressure(
    pressure,
    benzene_complex_system,
    toluene_complex_system,
    tmpdir,
    default_settings,
):
    """
    Test that the right nonbonded cutoff is propagated to the system.
    """
    default_settings.thermo_settings.pressure = pressure

    protocol = SepTopProtocol(
        settings=default_settings,
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
    with tmpdir.as_cwd():
        solv_setup_output = solv_setup_unit[0].run(dry=True)
        serialized_topology = solv_setup_output["topology"]
        serialized_system = solv_setup_output["system"]
        solv_sampler = sol_run_unit[0].run(
            serialized_system, serialized_topology, dry=True
        )["debug"]["sampler"]

        assert solv_sampler._thermodynamic_states[1].pressure == pressure


@pytest.mark.parametrize(
    "cutoff",
    [1.0 * offunit.nanometer, 12.0 * offunit.angstrom, 0.9 * offunit.nanometer],
)
def test_dry_run_ligand_system_cutoff(
    cutoff,
    benzene_complex_system,
    toluene_complex_system,
    tmpdir,
    default_settings,
):
    """
    Test that the right nonbonded cutoff is propagated to the system.
    """
    default_settings.solvent_solvation_settings.solvent_padding = (
        1.9 * offunit.nanometer
    )
    default_settings.forcefield_settings.nonbonded_cutoff = cutoff

    protocol = SepTopProtocol(
        settings=default_settings,
    )
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    dag_units = list(dag.protocol_units)
    # Only check the cutoff for the Solvent SetUp Unit
    solv_setup_unit = [u for u in dag_units if isinstance(u, SepTopSolventSetupUnit)]

    with tmpdir.as_cwd():
        serialized_system = solv_setup_unit[0].run(dry=True)["system"]
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
    tmpdir,
    default_settings,
):
    default_settings.protocol_repeats = 1
    default_settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    default_settings.solvent_solvation_settings.solvent_model = "tip4pew"
    default_settings.integrator_settings.reassign_velocities = True

    protocol = SepTopProtocol(settings=default_settings)

    # Create DAG from protocol, get the solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )

    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 4

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]
    sol_run_unit = [u for u in prot_units if isinstance(u, SepTopSolventRunUnit)]

    assert len(solv_setup_unit) == 1
    assert len(sol_run_unit) == 1

    with tmpdir.as_cwd():
        solv_setup_output = solv_setup_unit[0].run(dry=True)
        serialized_topology = solv_setup_output["topology"]
        serialized_system = solv_setup_output["system"]
        solv_run = sol_run_unit[0].run(
            serialized_system, serialized_topology, dry=True
        )["debug"]["sampler"]
        assert solv_run.is_periodic


def test_dry_run_benzene_toluene_noncubic(
    benzene_complex_system,
    toluene_complex_system,
    tmpdir,
    default_settings,
):
    default_settings.protocol_repeats = 1
    default_settings.solvent_solvation_settings.solvent_padding = (
        1.5 * offunit.nanometer
    )
    default_settings.solvent_solvation_settings.box_shape = "dodecahedron"

    protocol = SepTopProtocol(settings=default_settings)

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )

    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 4

    solv_setup_unit = [u for u in prot_units if isinstance(u, SepTopSolventSetupUnit)]

    assert len(solv_setup_unit) == 1

    with tmpdir.as_cwd():
        solv_setup_output = solv_setup_unit[0].run(dry=True)
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
    tmpdir,
    default_settings,
):
    """
    Create a test system with fictitious user supplied charges and
    ensure that they are properly passed through to the constructed
    alchemical system.
    """
    default_settings.protocol_repeats = 1

    protocol = SepTopProtocol(settings=default_settings)

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
    complex_setup_unit = [
        u for u in prot_units if isinstance(u, SepTopComplexSetupUnit)
    ]

    # check sol_unit charges
    with tmpdir.as_cwd():
        serialized_system = solv_setup_unit[0].run(dry=True)["system"]
        system = deserialize(serialized_system)
        nonbond = [
            f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
        ]
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
    with tmpdir.as_cwd():
        serialized_system = complex_setup_unit[0].run(dry=True)["system"]
        system = deserialize(serialized_system)
        nonbond = [
            f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
        ]
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
    tmpdir,
    default_settings,
):
    default_settings.protocol_repeats = 1
    default_settings.forcefield_settings.hydrogen_mass = 1.0
    default_settings.forcefield_settings.hydrogen_mass = 1.0

    protocol = SepTopProtocol(settings=default_settings)

    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    with tmpdir.as_cwd():
        errmsg = "too large for hydrogen mass"
        with pytest.raises(ValueError, match=errmsg):
            prot_units[0].run(dry=True)


@pytest.fixture
def T4L_xml(
    benzene_complex_system,
    toluene_complex_system,
    tmp_path_factory,
    default_settings,
):
    # Fixing the number of solvent molecules in the solvent settings
    # to test against reference xml
    default_settings.solvent_solvation_settings.solvent_padding = None
    default_settings.solvent_solvation_settings.number_of_solvent_molecules = 364
    protocol = SepTopProtocol(settings=default_settings)

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
    def test_particles(T4L_xml, T4L_reference_xml):
        nr_particles = T4L_xml.getNumParticles()
        nr_particles_ref = T4L_reference_xml.getNumParticles()
        assert nr_particles == nr_particles_ref
        particle_masses = [T4L_xml.getParticleMass(i) for i in range(nr_particles)]
        particle_masses_ref = [
            T4L_reference_xml.getParticleMass(i) for i in range(nr_particles)
        ]
        assert particle_masses

        for a, b in zip(particle_masses, particle_masses_ref):
            assert a == b

    @staticmethod
    def test_constraints(T4L_xml, T4L_reference_xml):
        nr_constraints = T4L_xml.getNumConstraints()
        nr_constraints_ref = T4L_reference_xml.getNumConstraints()
        assert nr_constraints == nr_constraints_ref
        constraints = [
            T4L_xml.getConstraintParameters(i) for i in range(nr_constraints)
        ]
        constraints_ref = [
            T4L_reference_xml.getConstraintParameters(i) for i in range(nr_constraints)
        ]
        assert constraints

        for a, b in zip(constraints, constraints_ref):
            # Particle 1
            assert a[0] == b[0]
            # Particle 2
            assert a[1] == b[1]
            # Constraint Quantity
            assert a[2] == b[2]


def test_unit_tagging(benzene_toluene_dag, tmpdir):
    # test that executing the units includes correct gen and repeat info
    dag_units = benzene_toluene_dag.protocol_units
    with (
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopComplexSetupUnit.run",
            return_value={
                "system": pathlib.Path("system.xml.bz2"),
                "topology": "topology.pdb",
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopComplexRunUnit._execute",
            return_value={
                "repeat_id": 0,
                "generation": 0,
                "simtype": "complex",
                "nc": "file.nc",
                "last_checkpoint": "chck.nc",
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopSolventSetupUnit.run",
            return_value={
                "system": pathlib.Path("system.xml.bz2"),
                "topology": "topology.pdb",
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopSolventRunUnit._execute",
            return_value={
                "repeat_id": 0,
                "generation": 0,
                "simtype": "solvent",
                "nc": "file.nc",
                "last_checkpoint": "chck.nc",
            },
        ),
    ):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)
    solv_repeats = set()
    complex_repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs["generation"] == 0
        if ret.outputs["simtype"] == "complex":
            complex_repeats.add(ret.outputs["repeat_id"])
        else:
            solv_repeats.add(ret.outputs["repeat_id"])
    # Repeat ids are random ints so just check their lengths
    assert len(complex_repeats) == len(solv_repeats) == 2


def test_gather(benzene_toluene_dag, tmpdir):
    # check that .gather behaves as expected
    with (
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopComplexSetupUnit.run",
            return_value={
                "system": pathlib.Path("system.xml.bz2"),
                "topology": "topology.pdb",
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopComplexRunUnit._execute",
            return_value={
                "repeat_id": 0,
                "generation": 0,
                "simtype": "complex",
                "nc": "file.nc",
                "last_checkpoint": "chck.nc",
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopSolventSetupUnit.run",
            return_value={
                "system": pathlib.Path("system.xml.bz2"),
                "topology": "topology.pdb",
            },
        ),
        mock.patch(
            "openfe.protocols.openmm_septop.equil_septop_method"
            ".SepTopSolventRunUnit._execute",
            return_value={
                "repeat_id": 0,
                "generation": 0,
                "simtype": "solvent",
                "nc": "file.nc",
                "last_checkpoint": "chck.nc",
            },
        ),
    ):
        dagres = gufe.protocols.execute_DAG(
            benzene_toluene_dag,
            shared_basedir=tmpdir,
            scratch_basedir=tmpdir,
            keep_shared=True,
        )

    protocol = SepTopProtocol(
        settings=SepTopProtocol.default_settings(),
    )

    res = protocol.gather([dagres])

    assert isinstance(res, openfe.protocols.openmm_septop.SepTopProtocolResult)


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, septop_json):
        d = json.loads(septop_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, septop_json):
        d = json.loads(septop_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = SepTopProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est
        assert est.m == pytest.approx(0.48, abs=0.5)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est.m == pytest.approx(0.0, abs=0.2)
        assert isinstance(est, offunit.Quantity)
        assert est.is_compatible_with(offunit.kilojoule_per_mole)

    def test_get_individual(self, protocolresult):
        inds = protocolresult.get_individual_estimates()

        assert isinstance(inds, dict)
        assert isinstance(inds["solvent"], list)
        assert isinstance(inds["complex"], list)
        assert len(inds["solvent"]) == len(inds["complex"]) == 1
        for e, u in itertools.chain(inds["solvent"], inds["complex"]):
            assert e.is_compatible_with(offunit.kilojoule_per_mole)
            assert u.is_compatible_with(offunit.kilojoule_per_mole)

    @pytest.mark.parametrize('key', ['solvent', 'complex'])
    def test_get_forwards_etc(self, key, protocolresult):
        far = protocolresult.get_forward_and_reverse_energy_analysis()

        assert isinstance(far, dict)
        assert isinstance(far[key], list)
        far1 = far[key][0]
        assert isinstance(far1, dict)

        for k in ['fractions', 'forward_DGs', 'forward_dDGs',
                  'reverse_DGs', 'reverse_dDGs']:
            assert k in far1

            if k == 'fractions':
                assert isinstance(far1[k], np.ndarray)

    @pytest.mark.parametrize('key', ['solvent', 'complex'])
    def test_get_frwd_reverse_none_return(self, key, protocolresult):
        # fetch the first result of type key
        data = [i for i in protocolresult.data[key].values()][0][0]
        # set the output to None
        data.outputs['forward_and_reverse_energies'] = None

        # now fetch the analysis results and expect a warning
        wmsg = ("were found in the forward and reverse dictionaries "
                f"of the repeats of the {key}")
        with pytest.warns(UserWarning, match=wmsg):
            protocolresult.get_forward_and_reverse_energy_analysis()

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_get_overlap_matrices(self, key, protocolresult):
        ovp = protocolresult.get_overlap_matrices()

        assert isinstance(ovp, dict)
        assert isinstance(ovp[key], list)
        assert len(ovp[key]) == 1

        ovp1 = ovp[key][0]
        assert isinstance(ovp1["matrix"], np.ndarray)
        if key == 'solvent':
            lambda_nr = 27
        else:
            lambda_nr = 19
        assert ovp1["matrix"].shape == (lambda_nr, lambda_nr)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_get_replica_transition_statistics(self, key, protocolresult):
        rpx = protocolresult.get_replica_transition_statistics()
        if key == 'solvent':
            lambda_nr = 27
        else:
            lambda_nr = 19
        assert isinstance(rpx, dict)
        assert isinstance(rpx[key], list)
        assert len(rpx[key]) == 1
        rpx1 = rpx[key][0]
        assert "eigenvalues" in rpx1
        assert "matrix" in rpx1

        assert rpx1["eigenvalues"].shape == (lambda_nr,)
        assert rpx1["matrix"].shape == (lambda_nr, lambda_nr)

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_equilibration_iterations(self, key, protocolresult):
        eq = protocolresult.equilibration_iterations()

        assert isinstance(eq, dict)
        assert isinstance(eq[key], list)
        assert len(eq[key]) == 1
        assert all(isinstance(v, float) for v in eq[key])

    @pytest.mark.parametrize("key", ["solvent", "complex"])
    def test_production_iterations(self, key, protocolresult):
        prod = protocolresult.production_iterations()

        assert isinstance(prod, dict)
        assert isinstance(prod[key], list)
        assert len(prod[key]) == 1
        assert all(isinstance(v, float) for v in prod[key])

    def test_filenotfound_replica_states(self, protocolresult):
        errmsg = "File could not be found"

        with pytest.raises(ValueError, match=errmsg):
            protocolresult.get_replica_states()
