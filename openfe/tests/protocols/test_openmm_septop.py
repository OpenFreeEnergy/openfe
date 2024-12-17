# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols.openmm_septop import SepTopProtocol
from openfe.protocols.openmm_septop.equil_septop_method import _check_alchemical_charge_difference
from openfe.protocols.openmm_utils import system_validation
import numpy
import openmm
import openmm.app
import openmm.unit

from openfe.protocols.openmm_septop.femto_utils import compute_energy, is_close
from openmmtools.alchemy import AlchemicalRegion, AbsoluteAlchemicalFactory


KJ_PER_MOL = openmm.unit.kilojoule_per_mole


E_CHARGE = 1.602176634e-19 * openmm.unit.coulomb
EPSILON0 = (
    1e-6
    * 8.8541878128e-12
    / (openmm.unit.AVOGADRO_CONSTANT_NA * E_CHARGE**2)
    * openmm.unit.farad
    / openmm.unit.meter
)
ONE_4PI_EPS0 = 1 / (4 * numpy.pi * EPSILON0) * EPSILON0.unit * 10.0  # nm -> angstrom


@pytest.fixture()
def default_settings():
    return SepTopProtocol.default_settings()


def test_create_default_settings():
    settings = SepTopProtocol.default_settings()
    assert settings


@pytest.mark.parametrize('val', [
    {'elec': [0.0, -1], 'vdw': [0.0, 1.0], 'restraints': [0.0, 1.0]},
    {'elec': [0.0, 1.5], 'vdw': [0.0, 1.5], 'restraints': [-0.1, 1.0]}
])
def test_incorrect_window_settings(val, default_settings):
    errmsg = "Lambda windows must be between 0 and 1."
    lambda_settings = default_settings.lambda_settings
    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_A = val['elec']
        lambda_settings.lambda_vdw_A = val['vdw']
        lambda_settings.lambda_restraints_A = val['restraints']


@pytest.mark.parametrize('val', [
    {'elec': [0.0, 0.1, 0.0], 'vdw': [0.0, 1.0, 1.0], 'restraints': [0.0, 1.0, 1.0]},
])
def test_monotonic_lambda_windows(val, default_settings):
    errmsg = "The lambda schedule is not monotonic."
    lambda_settings = default_settings.lambda_settings

    with pytest.raises(ValueError, match=errmsg):
        lambda_settings.lambda_elec_A = val['elec']
        lambda_settings.lambda_vdw_A = val['vdw']
        lambda_settings.lambda_restraints_A = val['restraints']


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0], 'vdw': [0.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nreplicas(val, default_settings):
    default_settings.lambda_settings.lambda_elec_A = val['elec']
    default_settings.lambda_settings.lambda_vdw_A = val['vdw']
    default_settings.lambda_settings.lambda_restraints_A = val['restraints']
    default_settings.lambda_settings.lambda_elec_B = val['elec']
    default_settings.lambda_settings.lambda_vdw_B = val['vdw']
    default_settings.lambda_settings.lambda_restraints_B = val[
        'restraints']
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (f"Number of replicas {n_replicas} does not equal the"
              f" number of lambda windows {len(val['vdw'])}")
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 1.0, 1.0], 'vdw': [0.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nwindows(val, default_settings):
    default_settings.lambda_settings.lambda_elec_A = val['elec']
    default_settings.lambda_settings.lambda_vdw_A = val['vdw']
    default_settings.lambda_settings.lambda_restraints_A = val['restraints']
    n_replicas = 3
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "Components elec, vdw, and restraints must have equal amount of lambda "
        "windows. Got 3 and 19 elec lambda windows")
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.complex_simulation_settings,
        )


@pytest.mark.parametrize('val', [
    {'elec': [1.0, 0.5], 'vdw': [1.0, 1.0], 'restraints': [0.0, 0.0]},
])
def test_validate_lambda_schedule_nakedcharge(val, default_settings):
    default_settings.lambda_settings.lambda_elec_A = val['elec']
    default_settings.lambda_settings.lambda_vdw_A = val['vdw']
    default_settings.lambda_settings.lambda_restraints_A = val[
        'restraints']
    default_settings.lambda_settings.lambda_elec_B = val['elec']
    default_settings.lambda_settings.lambda_vdw_B = val['vdw']
    default_settings.lambda_settings.lambda_restraints_B = val[
        'restraints']
    n_replicas = 2
    default_settings.complex_simulation_settings.n_replicas = n_replicas
    default_settings.solvent_simulation_settings.n_replicas = n_replicas
    errmsg = (
        "There are states along this lambda schedule "
        "where there are atoms with charges but no LJ "
        "interactions: Ligand A: l")
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.complex_simulation_settings,
        )
    with pytest.raises(ValueError, match=errmsg):
        SepTopProtocol._validate_lambda_schedule(
            default_settings.lambda_settings,
            default_settings.solvent_simulation_settings,
        )


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = SepTopProtocol(
        settings=default_settings,
    )
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = SepTopProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = SepTopProtocol.from_dict(ser)
    assert protocol == ret


def test_create_independent_repeat_ids(
        benzene_complex_system, toluene_complex_system,
):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = SepTopProtocol.default_settings()
    # Default protocol is 1 repeat, change to 3 repeats
    settings.protocol_repeats = 3
    protocol = SepTopProtocol(
            settings=settings,
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
        repeat_ids.add(u.inputs['repeat_id'])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])

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
        charged_benzene_modifications, T4_protein_component,
):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = SepTopProtocol.default_settings()
    # Default protocol is 1 repeat, change to 3 repeats
    settings.protocol_repeats = 3
    protocol = SepTopProtocol(
            settings=settings,
    )
    stateA = ChemicalSystem({
        'benzene': charged_benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzoic': charged_benzene_modifications['benzoic_acid'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })
    errmsg = "A charge difference of 1"
    with pytest.raises(ValueError, match=errmsg):
        protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )


def test_validate_complex_endstates_protcomp_stateA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    with pytest.raises(ValueError, match="No ProteinComponent found in stateA"):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)


def test_validate_complex_endstates_protcomp_stateB(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent(),
    })

    with pytest.raises(ValueError, match="No ProteinComponent found in stateB"):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)



def test_validate_complex_endstates_nosolvcomp_stateA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateA"
    ):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)


def test_validate_complex_endstates_nosolvcomp_stateB(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
    })

    with pytest.raises(
        ValueError, match="No SolventComponent found in stateB"
    ):
        SepTopProtocol._validate_complex_endstates(stateA, stateB)


def test_validate_alchem_comps_missingA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='one alchemical components must be present in stateA.'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_comps_missingB(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='one alchemical components must be present in stateB.'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_comps_toomanyA(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'toluene': benzene_modifications['toluene'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    stateB = ChemicalSystem({
        'phenol': benzene_modifications['phenol'],
        'protein': T4_protein_component,
        'solvent': SolventComponent(),
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    assert len(alchem_comps['stateA']) == 2

    assert len(alchem_comps['stateB']) == 1

    with pytest.raises(ValueError, match='Found 2 alchemical components in stateA'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


def test_validate_alchem_nonsmc(
    benzene_modifications, T4_protein_component,
):
    stateA = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'solvent': SolventComponent()
    })

    stateB = ChemicalSystem({
        'benzene': benzene_modifications['benzene'],
        'protein': T4_protein_component
    })

    alchem_comps = system_validation.get_alchemical_components(stateA, stateB)

    with pytest.raises(ValueError, match='Non SmallMoleculeComponent'):
        SepTopProtocol._validate_alchemical_components(alchem_comps)


### Tests for the alchemical systems. This tests were modified from
### femto (https://github.com/Psivant/femto/tree/main)
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
    ) * KJ_PER_MOL


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
        epsilon = numpy.sqrt(epsilons[idx_a] * epsilons[idx_b])
        sigma = 0.5 * (sigmas[idx_a] + sigmas[idx_b])
        charge = charges[idx_a] * charges[idx_b]

        return compute_interaction_energy(
            epsilon, sigma, charge, distances[idx_a][idx_b], lambda_vdw, lambda_charges
        )

    coords = (
        numpy.array(
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
        alchemical_region_A = AlchemicalRegion(
            alchemical_atoms=[0], name='A')
        alchemical_system = factory.create_alchemical_system(
            system, [alchemical_region_A])

        energy_0 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 1.0,
                'lambda_electrostatics_A': 1.0,
            },
        )

        # expect lig_1 + solvent, lig_1 + lig_2 and lig_2 + solvent
        # interaction when
        # lambda=0
        expected_energy_0 = energy_fn(0, 2) + energy_fn(0, 1) + energy_fn(1, 2)
        assert is_close(energy_0, expected_energy_0)

        # expect only lig_2 + solvent interaction when lambda=1
        energy_1 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 0.0,
                'lambda_electrostatics_A': 0.0,
            },
        )
        expected_energy_1 = energy_fn(1, 2)
        assert is_close(energy_1, expected_energy_1)

        # expect all particles to interact but only lig - solvent interactions to be
        # scaled
        energy_05 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 0.5,
                'lambda_electrostatics_A': 0.5,
            },
        )
        expected_energy_05 = (
            energy_fn(1, 2) + energy_fn(0, 2, 0.5, 0.5) + energy_fn(0, 1, 0.5, 0.5)
        )
        assert is_close(energy_05, expected_energy_05)

    def test_two_ligands(self, three_particle_system):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system

        # Do it the openmm way
        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(
            alchemical_atoms=[0], name='A')
        alchemical_region_B = AlchemicalRegion(
            alchemical_atoms=[1], name='B')
        alchemical_system = factory.create_alchemical_system(
            system, [alchemical_region_A, alchemical_region_B])
        energy_0 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 1.0,
                'lambda_electrostatics_A': 1.0,
                'lambda_sterics_B': 0.0,
                'lambda_electrostatics_B': 0.0,
            },
        )

        # expect only lig_1 + solvent interaction when lambda=0
        expected_energy_0 = energy_fn(0, 2)
        assert is_close(energy_0, expected_energy_0)

        # expect only lig_2 + solvent interaction when lambda=1
        energy_1 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 0.0,
                'lambda_electrostatics_A': 0.0,
                'lambda_sterics_B': 1.0,
                'lambda_electrostatics_B': 1.0,
            },
        )
        expected_energy_1 = energy_fn(1, 2)
        assert is_close(energy_1, expected_energy_1)

        # expect lig_1 + solvent and lig_2 + solvent interaction when lambda=0.5
        # but no lig_1 + lig_2 interaction by default
        energy_05 = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 0.5,
                'lambda_electrostatics_A': 0.5,
                'lambda_sterics_B': 0.5,
                'lambda_electrostatics_B': 0.5,
            },
        )
        expected_energy_05 = energy_fn(0, 2, 0.5, 0.5) + energy_fn(1, 2, 0.5, 0.5)
        assert is_close(energy_05, expected_energy_05)

    def test_two_ligands_charges(self, three_particle_system):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system

        # Do it the openmm way
        factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
        alchemical_region_A = AlchemicalRegion(
            alchemical_atoms=[0], name='A')
        alchemical_region_B = AlchemicalRegion(
            alchemical_atoms=[1], name='B')
        alchemical_system = factory.create_alchemical_system(
            system, [alchemical_region_A, alchemical_region_B])
        energy = compute_energy(
            alchemical_system,
            coords,
            None,
            {
                'lambda_sterics_A': 1.0,
                'lambda_electrostatics_A': 0.8,
                'lambda_sterics_B': 1.0,
                'lambda_electrostatics_B': 0.2,
            },
        )
        expected_energy = energy_fn(0, 2, 1.0, 0.8) + energy_fn(1, 2, 1.0, 0.2)
        assert is_close(energy, expected_energy)
