# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import itertools
import json
import sys
from math import sqrt
from unittest import mock

import gufe
import mdtraj as mdt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm
from openmm import (
    CustomBondForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    MonteCarloBarostat,
    NonbondedForce,
    PeriodicTorsionForce,
)
from openmmtools.multistate.multistatesampler import MultiStateSampler

import openfe
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteSolvationProtocol,
    AHFESolventSetupUnit,
    AHFEVacuumSetupUnit,
    AHFESolventSimUnit,
    AHFEVacuumSimUnit,
    AHFESolventAnalysisUnit,
    AHFEVacuumAnalysisUnit,
)
from openfe.protocols.openmm_utils import system_validation
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_ESPALOMA_CHARGE,
    HAS_NAGL,
    HAS_OPENEYE,
)


UNIT_TYPES = {
    'solvent': {
        'setup': AHFESolventSetupUnit,
        'sim': AHFESolventSimUnit,
        'analysis': AHFESolventAnalysisUnit,
    },
    'vacuum': {
        'setup': AHFEVacuumSetupUnit,
        'sim': AHFEVacuumSimUnit,
        'analysis': AHFEVacuumAnalysisUnit,
    }
}


def _get_units(protocol_units, unit_type):
    """
    Helper method to extract setup units.
    """
    return [pu for pu in protocol_units if isinstance(pu, unit_type)]


@pytest.fixture()
def protocol_dry_settings():
    settings = AbsoluteSolvationProtocol.default_settings()
    settings.vacuum_engine_settings.compute_platform = None
    settings.solvent_engine_settings.compute_platform = None
    settings.protocol_repeats = 1
    return settings


@pytest.fixture()
def default_settings():
    return AbsoluteSolvationProtocol.default_settings()


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = AbsoluteSolvationProtocol(settings=default_settings)
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = AbsoluteSolvationProtocol(settings=default_settings)

    ser = protocol.to_dict()
    ret = AbsoluteSolvationProtocol.from_dict(ser)
    assert protocol == ret


def test_repeat_units(benzene_system):
    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=opemm_afe.AbsoluteSolvationProtocol.default_settings()
    )

    dag = protocol.create(
        stateA=benzene_system,
        stateB=ChemicalSystem({'solvent': SolventComponent()}),
    )

    # 9 protocol units, 3 per repeat
    pus = list(dag.protocol_units)
    assert len(pus) == 9

    # Check info for each repeat
    for phase in ['solvent', 'vacuum']:
        setup = _get_units(pus, UNIT_TYPES[phase]['setup'])
        sim = _get_units(pus, UNIT_TYPES[phase]['sim'])
        analysis = _get_units(pus, UNIT_TYPES[phase]['analysis'])

        # Should be 3 of each set
        assert len(setup) == len(sim) == len(analysis) == 3

        # Check that the dag chain is correct
        for analysis_pu in analysis:
            repeat_id = analysis_pu.inputs["repeat_id"]
            setup_pu = [s for s in setup if s.inputs["repeat_id"] == repeat_id][0]
            sim_pu = [s for s in sim if s.inputs["repeat_id"] == repeat_id][0]
            assert analysis_pu.inputs["setup_results"] == setup_pu
            assert analysis_pu.inputs["simulation_reuslts"] == sim_pu
            assert sim_pu.inputs["setup_results"] == setup_pu


def test_create_independent_repeat_ids(benzene_system):
    protocol = openmm_afe.AbsoluteSolvationProtocol(
        settings=openmm_afe.AbsoluteSolvationProtocol.default_settings()
    )

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    dags = []
    for i in range(2):
        dags.append(
            protocol.create(
                stateA=benzene_system,
                stateB=stateB,
                mapping=None,
            )
        )

    repeat_ids = set()

    for dag in dags:
        # 3 sets of 3 units
        assert len(list(dag.protocol_units)) == 9
        for u in dag.protocol_units:
            repeat_ids.add(u.inputs["repeat_id"])

    assert len(repeat_ids) == 12


def _assert_num_forces(system, forcetype, number):
    """
    Helper method to check the number of forces of a given
    type in a system.
    """
    forces = [f for f in system.getForces() if isinstance(f, forcetype)]
    assert len(forces) == number


def _verify_alchemical_sterics_force_parameters(
    force,
    long_range=True,
    alpha=0.5,
    beta=0,
    a=1.0,
    b=1.0,
    c=6.0,
    d=1.0,
    e=1.0,
    f=2.0,
):
    assert force.getUseLongRangeCorrection() is long_range

    if force.getNumGlobalParameters() == 8:
        shift = 0
    else:
        shift = 1

    # Check the softcore parameters for the sterics forces
    assert force.getGlobalParameterName(0 + shift) == "softcore_alpha"
    assert force.getGlobalParameterName(1 + shift) == "softcore_beta"
    assert force.getGlobalParameterName(2 + shift) == "softcore_a"
    assert force.getGlobalParameterName(3 + shift) == "softcore_b"
    assert force.getGlobalParameterName(4 + shift) == "softcore_c"
    assert force.getGlobalParameterName(5 + shift) == "softcore_d"
    assert force.getGlobalParameterName(6 + shift) == "softcore_e"
    assert force.getGlobalParameterName(7 + shift) == "softcore_f"

    assert force.getGlobalParameterDefaultValue(0 + shift) == alpha
    assert force.getGlobalParameterDefaultValue(1 + shift) == beta
    assert force.getGlobalParameterDefaultValue(2 + shift) == a
    assert force.getGlobalParameterDefaultValue(3 + shift) == b
    assert force.getGlobalParameterDefaultValue(4 + shift) == c
    assert force.getGlobalParameterDefaultValue(5 + shift) == d
    assert force.getGlobalParameterDefaultValue(6 + shift) == e
    assert force.getGlobalParameterDefaultValue(7 + shift) == f


@pytest.mark.parametrize("method", ["repex", "sams", "independent", "InDePeNdENT"])
def test_setup_dry_sim_vac_benzene(benzene_system, method, protocol_dry_settings, tmpdir):
    protocol_dry_settings.vacuum_simulation_settings.sampler_method = method

    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 6

    vac_setup_unit = _get_units(prot_units, UNIT_TYPES['vacuum']['setup'])
    vac_sim_unit = _get_units(prot_units, UNIT_TYPES['vacuum']['sim'])

    assert len(vac_setup_unit) == 1
    assert len(vac_sim_unit) == 1

    with tmpdir.as_cwd():
        setup_results = vac_setup_unit[0].run(dry=True)
        sim_results = vac_sim_unit[0].run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            dry=True,
        )

        sampler = sim_results["sampler"]
        assert isinstance(sampler, MultiStateSampler)
        assert not sampler.is_periodic
        assert sampler._thermodynamic_states[0].barostat is None

        # standard system
        system = setup_results["standard_system"]
        assert system.getNumParticles() == 12
        assert len(system.getForces()) == 4
        _assert_num_forces(system, NonbondedForce, 1)
        _assert_num_forces(system, HarmonicBondForce, 1)
        _assert_num_forces(system, HarmonicAngleForce, 1)
        _assert_num_forces(system, PeriodicTorsionForce, 1)

        # alchemical system
        alchem_system = setup_results["alchem_system"]
        assert alchem_system.getNumParticles() == 12
        assert len(alchem_system.getForces()) == 12
        _assert_num_forces(alchem_system, NonbondedForce, 1)
        _assert_num_forces(alchem_system, CustomNonbondedForce, 4)
        _assert_num_forces(alchem_system, CustomBondForce, 4)
        _assert_num_forces(alchem_system, HarmonicBondForce, 1)
        _assert_num_forces(alchem_system, HarmonicAngleForce, 1)
        _assert_num_forces(alchem_system, PeriodicTorsionForce, 1)

        # Check some force contents
        stericsf = [
            f
            for f in alchem_system.getForces()
            if isinstance(f, CustomNonbondedForce) and "U_sterics" in f.getEnergyFunction()
        ]

        for force in stericsf:
            _verify_alchemical_sterics_force_parameters(force)


@pytest.mark.parametrize(
    "alpha, a, b, c, correction",
    [
        [0.2, 2, 2, 1, True],
        [0.35, 2.2, 1.5, 0, False],
    ],
)
def test_alchemical_settings_setup_vacuum(
    alpha, a, b, c, correction, benzene_system, protocol_dry_settings, tmpdir
):
    """
    Test non default alchemical settings
    """
    protocol_dry_settings.alchemical_settings.softcore_alpha = alpha
    protocol_dry_settings.alchemical_settings.softcore_a = a
    protocol_dry_settings.alchemical_settings.softcore_b = b
    protocol_dry_settings.alchemical_settings.softcore_c = c
    protocol_dry_settings.alchemical_settings.disable_alchemical_dispersion_correction = correction
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 6

    vac_setup_unit = _get_units(prot_units, UNIT_TYPES['vacuum']['setup'])
    vac_sim_unit = _get_units(prot_units, UNIT_TYPES['vacuum']['sim'])

    assert len(vac_setup_unit) == 1
    assert len(vac_sim_unit) == 1

    with tmpdir.as_cwd():
        results = vac_setup_unit[0].run(dry=True)

        alchem_system = results["alchem_system"]
        _assert_num_forces(alchem_system, NonbondedForce, 1)
        _assert_num_forces(alchem_system, CustomNonbondedForce, 4)
        _assert_num_forces(alchem_system, CustomBondForce, 4)
        _assert_num_forces(alchem_system, HarmonicBondForce, 1)
        _assert_num_forces(alchem_system, HarmonicAngleForce, 1)
        _assert_num_forces(alchem_system, PeriodicTorsionForce, 1)

        # Check some force contents
        stericsf = [
            f
            for f in alchem_system.getForces()
            if isinstance(f, CustomNonbondedForce) and "U_sterics" in f.getEnergyFunction()
        ]

        for force in stericsf:
            _verify_alchemical_sterics_force_parameters(
                force,
                long_range=not correction,
                alpha=alpha,
                a=a,
                b=b,
                c=c,
            )


def test_confgen_fail_AFE(benzene_system, protocol_dry_settings, tmpdir):
    # check system parametrisation works even if confgen fails
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first vacuum unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)
    vac_setup_unit = _get_units(prot_units, UNIT_TYPES['vacuum']['setup'])

    with tmpdir.as_cwd():
        with mock.patch("rdkit.Chem.AllChem.EmbedMultipleConfs", return_value=0):
            vac_sampler = vac_setup_unit[0].run(dry=True)["sampler"]
            assert vac_sampler


def test_setup_solv_benzene(benzene_system, protocol_dry_settings, tmpdir):
    protocol_dry_settings.solvent_output_settings.output_indices = "resname UNK"

    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_unit = _get_units(prot_units, UNIT_TYPES['solvent']['setup'])

    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        results = sol_unit[0].run(dry=True)
        sol_sampler = results["sampler"]
        assert sol_sampler.is_periodic

        pdb = mdt.load_pdb(results["pdb_structure"])
        assert pdb.n_atoms == 12


def test_dry_run_vsite_fail(benzene_system, tmpdir, protocol_dry_settings):
    protocol_dry_settings.vacuum_forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    protocol_dry_settings.solvent_forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    protocol_dry_settings.solvation_settings.solvent_model = "tip4pew"
    protocol_dry_settings.integrator_settings.reassign_velocities = False

    protocol = AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_setup_unit = _get_units(prot_units, UNIT_TYPES['solvent']['setup'])
    sol_sim_unit = _get_units(prot_units, UNIT_TYPES['solvent']['sim'])

    with tmpdir.as_cwd():
        setup_results = sol_setup_unit[0].run(dry=True)
        with pytest.raises(ValueError, match="are unstable"):
            sim_results = sol_sim_unit[0].run(
                system=setup_results["alchem_system"],
                positions=setup_results["debug_positions"],
                selection_indices=setup_results["selection_indices"],
                box_vectors=setup_results["box_vectors"],
                alchemical_restraints=False,
                dry=True,
            )


def test_setup_dry_sim_solv_benzene_tip4p(
    benzene_system, protocol_dry_settings, tmpdir
):
    protocol_dry_settings.vacuum_forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    protocol_dry_settings.solvent_forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    protocol_dry_settings.solvation_settings.solvent_model = "tip4pew"
    protocol_dry_settings.integrator_settings.reassign_velocities = True

    protocol = AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_setup_unit = _get_units(prot_units, UNIT_TYPES['solvent']['setup'])     sol_sim_unit = _get_units(prot_units, UNIT_TYPES['solvent']['sim'])

    with tmpdir.as_cwd():
        setup_results = sol_setup_unit[0].run(dry=True)
        sim_results = sol_sim_unit[0].run(
            system=setup_results["alchem_system"],
            positions=setup_results["debug_positions"],
            selection_indices=setup_results["selection_indices"],
            box_vectors=setup_results["box_vectors"],
            alchemical_restraints=False,
            dry=True,
        )
        sol_sampler = sim_results["sampler"]
        assert sol_sampler.is_periodic


def test_dry_run_solv_benzene_noncubic(benzene_system, protocol_dry_settings, tmpdir):
    protocol_dry_settings.solvation_settings.solvent_padding = 1.5 * offunit.nanometer
    protocol_dry_settings.solvation_settings.box_shape = "dodecahedron"

    protocol = AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    sol_setup_unit = _get_units(prot_units, UNIT_TYPES['solvent']['setup'])

    with tmpdir.as_cwd():
        results = sol_setup_unit[0].run(dry=True)
        system = results["alchem_system"]

        vectors = system.getDefaultPeriodicBoxVectors()
        width = float(from_openmm(vectors)[0][0].to("nanometer").m)

        # dodecahedron has the following shape:
        # [width, 0, 0], [0, width, 0], [0.5, 0.5, 0.5 * sqrt(2)] * width

        expected_vectors = [
            [width, 0, 0],
            [0, width, 0],
            [0.5 * width, 0.5 * width, 0.5 * sqrt(2) * width],
        ] * offunit.nanometer
        assert_allclose(expected_vectors, from_openmm(vectors))


def test_dry_run_solv_user_charges_benzene(
    benzene_modifications, protocol_dry_settings, tmpdir
):
    """
    Create a test system with fictitious user supplied charges and
    ensure that they are properly passed through to the constructed
    alchemical system.
    """
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    def assign_fictitious_charges(offmol):
        """
        Get a random array of fake partial charges for your offmol.
        """
        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
        rand_arr[-1] = -sum(rand_arr[:-1])
        return rand_arr * offunit.elementary_charge

    benzene_offmol = benzene_modifications["benzene"].to_openff()
    offmol_pchgs = assign_fictitious_charges(benzene_offmol)
    benzene_offmol.partial_charges = offmol_pchgs
    benzene_smc = openfe.SmallMoleculeComponent.from_openff(benzene_offmol)

    # check propchgs
    prop_chgs = benzene_smc.to_dict()["molprops"]["atom.dprop.PartialCharge"]
    prop_chgs = np.array(prop_chgs.split(), dtype=float)
    np.testing.assert_allclose(prop_chgs, offmol_pchgs)

    # Create ChemicalSystems
    stateA = ChemicalSystem(
        {
            "benzene": benzene_smc,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None)
    prot_units = list(dag.protocol_units)

    vac_setup_units = _get_units(prot_units, UNIT_TYPES['vacuum']['setup'])
    sol_setup_units = _get_untis(prot_units, UNIT_TYPES['solvent']['setup'])

    # check sol_unit charges
    with tmpdir.as_cwd():
        results = sol_setup_unit.run(dry=True)
        system = results["alchem_system"]
        nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)]

        assert len(nonbond) == 1

        # loop through the 12 benzene atoms
        # partial charge is stored in the offset
        for i in range(12):
            offsets = nonbond[0].getParticleParameterOffset(i)
            c = ensure_quantity(offsets[2], "openff")
            assert pytest.approx(c) == prop_chgs[i]

    # check vac_unit charges
    with tmpdir.as_cwd():
        results = vac_setup_unit.run(dry=True)
        system = results["alchem_system"]
        nonbond = [f for f in system.getForces() if isinstance(f, CustomNonbondedForce)]
        assert len(nonbond) == 4

        custom_elec = [
            n for n in nonbond if n.getGlobalParameterName(0) == "lambda_electrostatics"
        ][0]

        # loop through the 12 benzene atoms
        for i in range(12):
            c, s = custom_elec.getParticleParameters(i)
            c = ensure_quantity(c, "openff")
            assert pytest.approx(c) == prop_chgs[i]


@pytest.mark.parametrize(
    "method, backend, ref_key",
    [
        ("am1bcc", "ambertools", "ambertools"),
        pytest.param(
            "am1bcc",
            "openeye",
            "openeye",
            marks=pytest.mark.skipif(not HAS_OPENEYE, reason="needs oechem"),
        ),
        pytest.param(
            "nagl",
            "rdkit",
            "nagl",
            marks=pytest.mark.skipif(
                not HAS_NAGL or sys.platform.startswith("darwin"),
                reason="needs NAGL and/or on macos",
            ),
        ),
        pytest.param(
            "espaloma",
            "rdkit",
            "espaloma",
            marks=pytest.mark.skipif(not HAS_ESPALOMA_CHARGE, reason="needs espaloma charge"),
        ),
    ],
)
def test_dry_run_charge_backends(
    CN_molecule, tmpdir, method, backend, ref_key, protocol_dry_settings, am1bcc_ref_charges
):
    """
    Check that partial charge generation with different backends
    works as expected.
    """
    protocol_dry_settings.partial_charge_settings.partial_charge_method = method
    protocol_dry_settings.partial_charge_settings.off_toolkit_backend = backend
    protocol_dry_settings.partial_charge_settings.nagl_model = "openff-gnn-am1bcc-0.1.0-rc.1.pt"

    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    # Create ChemicalSystems
    stateA = ChemicalSystem({"benzene": CN_molecule, "solvent": SolventComponent()})

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None)
    prot_units = list(dag.protocol_units)

    vac_setup_units = _get_units(prot_units, UNIT_TYPES['vacuum']['setup'])

    # check vac_unit charges
    with tmpdir.as_cwd():
        results = vac_setup_units[0].run(dry=True)
        system = results["alchem_system"]
        nonbond = [f for f in system.getForces() if isinstance(f, CustomNonbondedForce)]
        assert len(nonbond) == 4

        custom_elec = [
            n for n in nonbond if n.getGlobalParameterName(0) == "lambda_electrostatics"
        ][0]

        charges = []
        for i in range(system.getNumParticles()):
            c, s = custom_elec.getParticleParameters(i)
            charges.append(c)

    assert_allclose(
        am1bcc_ref_charges[ref_key],
        charges * offunit.elementary_charge,
        rtol=1e-4,
    )


@pytest.fixture
def benzene_solvation_dag(benzene_system, protocol_dry_settings):
    protocol_dry_settings.protocol_repeats = 3
    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    return protocol.create(stateA=stateA, stateB=stateB, mapping=None)


def test_unit_tagging(benzene_solvation_dag, tmpdir):
    # test that executing the units includes correct gen and repeat info

    dag_units = benzene_solvation_dag.protocol_units

    with (
        mock.patch(
            "openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationSolventUnit.run",
            return_value={"nc": "file.nc", "last_checkpoint": "chck.nc"},
        ),
        mock.patch(
            "openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationVacuumUnit.run",
            return_value={"nc": "file.nc", "last_checkpoint": "chck.nc"},
        ),
    ):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    solv_repeats = set()
    vac_repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs["generation"] == 0
        if ret.outputs["simtype"] == "vacuum":
            vac_repeats.add(ret.outputs["repeat_id"])
        else:
            solv_repeats.add(ret.outputs["repeat_id"])
    # Repeat ids are random ints so just check their lengths
    assert len(vac_repeats) == len(solv_repeats) == 3


@pytest.mark.parametrize(
    "positions_write_frequency,velocities_write_frequency",
    [
        [100 * offunit.picosecond, None],
        [None, None],
        [None, 100 * offunit.picosecond],
    ],
)
def test_dry_run_vacuum_write_frequency(
    benzene_system,
    positions_write_frequency,
    velocities_write_frequency,
    protocol_dry_settings,
    tmpdir,
):
    protocol_dry_settings.solvent_output_settings.output_indices = "resname UNK"
    protocol_dry_settings.solvent_output_settings.positions_write_frequency = positions_write_frequency  # fmt: skip
    protocol_dry_settings.solvent_output_settings.velocities_write_frequency = velocities_write_frequency  # fmt: skip
    protocol_dry_settings.vacuum_output_settings.positions_write_frequency = positions_write_frequency  # fmt: skip
    protocol_dry_settings.vacuum_output_settings.velocities_write_frequency = velocities_write_frequency  # fmt: skip
    # set the time per iteration to 1 ps to make the math easy
    protocol_dry_settings.solvent_simulation_settings.time_per_iteration = 1 * offunit.picosecond
    protocol_dry_settings.vacuum_simulation_settings.time_per_iteration = 1 * offunit.picosecond

    protocol = openmm_afe.AbsoluteSolvationProtocol(settings=protocol_dry_settings)

    stateA = benzene_system

    stateB = ChemicalSystem({"solvent": SolventComponent()})

    # Create DAG from protocol, get the vacuum and solvent units
    # and eventually dry run the first solvent unit
    dag = protocol.create(
        stateA=stateA,
        stateB=stateB,
        mapping=None,
    )
    prot_units = list(dag.protocol_units)

    assert len(prot_units) == 2

    with tmpdir.as_cwd():
        for u in prot_units:
            sampler = u.run(dry=True)["debug"]["sampler"]
            reporter = sampler._reporter
            if positions_write_frequency:
                assert reporter.position_interval == positions_write_frequency.m
            else:
                assert reporter.position_interval == 0
            if velocities_write_frequency:
                assert reporter.velocity_interval == velocities_write_frequency.m
            else:
                assert reporter.velocity_interval == 0
