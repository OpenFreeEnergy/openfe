# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import json
import logging
import pathlib
import sys
from unittest import mock

import gufe
import numpy as np
import openmm
import pytest
from gufe import ChemicalSystem, LigandAtomMapping, SmallMoleculeComponent
from numpy.testing import assert_allclose
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import MonteCarloBarostat, NonbondedForce
from openmm import unit as omm_unit
from openmmtools.states import ThermodynamicState
from pydantic import ValidationError

import openfe
from openfe.protocols import openmm_md
from openfe.protocols.openmm_md.plain_md_methods import (
    PlainMDProtocol,
    PlainMDProtocolResult,
    PlainMDSetupUnit,
    PlainMDSimulationUnit,
)
from openfe.protocols.openmm_utils import serialization
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_ESPALOMA_CHARGE,
    HAS_NAGL,
    HAS_OPENEYE,
)
from openfe.tests.conftest import HAS_ESPALOMA


@pytest.fixture()
def vac_settings():
    settings = PlainMDProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    settings.engine_settings.compute_platform = None
    return settings


@pytest.mark.parametrize(
    "inputs, expected",
    [
        # inputs are current step count, nvt steps, npt steps and prod steps
        # outputs are steps to run for nvt, npt, prod and if the production phase has started
        pytest.param([50, 100, 100, 100], [50, 100, 100, False], id="nvt resuming"),
        pytest.param([101, 100, 100, 100], [0, 99, 100, False], id="npt resuming"),
        pytest.param([220, 100, 100, 100], [0, 0, 80, True], id="prod resuming"),
        pytest.param([200, 100, 100, 100], [0, 0, 100, False], id="prod resuming not started"),
    ],
)
def test_get_remaining_steps(inputs, expected):
    nvt, npt, prod, is_prod = PlainMDSimulationUnit._get_remaining_steps(*inputs)
    assert nvt == expected[0]
    assert npt == expected[1]
    assert prod == expected[2]
    assert is_prod == expected[3]


def test_create_default_settings():
    settings = PlainMDProtocol.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = PlainMDProtocol(
        settings=PlainMDProtocol.default_settings(),
    )

    assert protocol


def test_invalid_protocol_repeats():
    settings = PlainMDProtocol.default_settings()
    with pytest.raises(ValueError, match="must be a positive value"):
        settings.protocol_repeats = -1


def test_serialize_protocol():
    protocol = PlainMDProtocol(
        settings=PlainMDProtocol.default_settings(),
    )

    ser = protocol.to_dict()

    ret = PlainMDProtocol.from_dict(ser)

    assert protocol == ret


def test_create_independent_repeat_ids(benzene_system):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = PlainMDProtocol.default_settings()
    # Default protocol is 1 repeat, change to 3 repeats
    settings.protocol_repeats = 3
    protocol = PlainMDProtocol(settings=settings)
    dag1 = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag2 = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )

    repeat_ids = set()
    u: PlainMDSetupUnit | PlainMDSimulationUnit
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])

    assert len(repeat_ids) == 6


def test_dry_run_default_vacuum(benzene_vacuum_system, vac_settings, tmp_path):
    protocol = PlainMDProtocol(settings=vac_settings)

    # create DAG from protocol and take the setup unit
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    setup_unit = list(dag.protocol_units)[0]
    result = setup_unit.run(
        dry=True, verbose=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )
    system = result["debug"]["system"]
    assert not ThermodynamicState(
        system,
        temperature=to_openmm(protocol.settings.thermo_settings.temperature),
    ).is_periodic

    assert (
        ThermodynamicState(
            system,
            temperature=to_openmm(protocol.settings.thermo_settings.temperature),
        ).barostat
        is None
    )


def test_dry_run_logger_output(benzene_vacuum_system, vac_settings, tmp_path, caplog):
    vac_settings.simulation_settings.equilibration_length_nvt = 1 * unit.picosecond
    vac_settings.simulation_settings.equilibration_length = 1 * unit.picosecond
    vac_settings.simulation_settings.production_length = 1 * unit.picosecond

    protocol = PlainMDProtocol(
        settings=vac_settings,
    )

    # create DAG from protocol
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    setup_unit = list(dag.protocol_units)[0]

    caplog.set_level(logging.INFO)
    setup_results = setup_unit.run(
        dry=False, verbose=True, scratch_basepath=tmp_path, shared_basepath=tmp_path
    )

    messages = [r.message for r in caplog.records]
    assert "Creating system" in messages
    # now run the production unit after extracting outputs from the setup unit
    system = serialization.deserialize(setup_results["system"])
    positions = np.load(setup_results["positions"]) * omm_unit.nanometers
    topology = openmm.app.PDBFile(str(setup_results["system_pdb"])).getTopology()
    equil_steps_nvt = setup_results["equil_steps_nvt"]
    equil_steps_npt = setup_results["equil_steps_npt"]
    prod_steps = setup_results["prod_steps"]
    prod_unit = list(dag.protocol_units)[1]
    prod_unit.run(
        system=system,
        positions=positions,
        topology=topology,
        equil_steps_nvt=equil_steps_nvt,
        equil_steps_npt=equil_steps_npt,
        prod_steps=prod_steps,
        dry=False,
        verbose=True,
        scratch_basepath=tmp_path,
        shared_basepath=tmp_path,
    )
    messages = [r.message for r in caplog.records]
    assert "Minimizing systems" in messages
    assert "Running NVT equilibration for 250 steps" in messages
    assert "Running NPT equilibration for 250 steps" in messages
    assert "Running production phase for 250 steps" in messages


def test_dry_run_ffcache_none_vacuum(benzene_vacuum_system, vac_settings, tmp_path):
    vac_settings.output_settings.forcefield_cache = None

    protocol = PlainMDProtocol(
        settings=vac_settings,
    )
    assert protocol.settings.output_settings.forcefield_cache is None

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]
    dag_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)["debug"]["system"]


def test_dry_run_gaff_vacuum(benzene_vacuum_system, vac_settings, tmp_path):
    vac_settings.forcefield_settings.small_molecule_forcefield = "gaff-2.11"

    protocol = PlainMDProtocol(
        settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]
    dag_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)["debug"]["system"]


@pytest.mark.xfail(reason="Issue #1940")
@pytest.mark.skipif(not HAS_ESPALOMA, reason="espaloma is not available")
def test_dry_run_espaloma_vacuum_user_charges(benzene_modifications, vac_settings, tmp_path):
    vac_settings.forcefield_settings.small_molecule_forcefield = "espaloma-0.3.2"

    protocol = PlainMDProtocol(
        settings=vac_settings,
    )

    # add some dummy charges to the benzene molecule
    benzene = benzene_modifications["benzene"]
    benzene_openff = benzene.to_openff()
    # assign some fake charges
    expected_charges = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    benzene_openff.partial_charges = expected_charges * unit.elementary_charge
    benzene_system = ChemicalSystem({"ligand": SmallMoleculeComponent.from_openff(benzene_openff)})
    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]
    result = dag_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)
    system = result["debug"]["system"]
    assert system.getNumParticles() == 12
    # check the charges assigned
    nb_force = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    charges = []
    for i in range(nb_force.getNumParticles()):
        c, _, _ = nb_force.getParticleParameters(i)
        charges.append(c.value_in_unit(omm_unit.elementary_charge))
    assert_allclose(charges, expected_charges, rtol=1e-6)


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
                not HAS_NAGL or HAS_OPENEYE or sys.platform.startswith("darwin"),
                reason="needs NAGL (without oechem) and/or on macos",
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
    CN_molecule, tmp_path, method, backend, ref_key, vac_settings, am1bcc_ref_charges
):
    vac_settings.partial_charge_settings.partial_charge_method = method
    vac_settings.partial_charge_settings.off_toolkit_backend = backend
    vac_settings.partial_charge_settings.nagl_model = "openff-gnn-am1bcc-0.1.0-rc.1.pt"

    protocol = PlainMDProtocol(settings=vac_settings)

    csystem = openfe.ChemicalSystem({"ligand": CN_molecule})

    dag = protocol.create(stateA=csystem, stateB=csystem, mapping=None)
    md_unit = list(dag.protocol_units)[0]

    result = md_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)
    system = result["debug"]["system"]

    nonbond = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]

    charges = []
    for i in range(system.getNumParticles()):
        c, s, e = nonbond.getParticleParameters(i)
        charges.append(from_openmm(c))

    charges = unit.Quantity.from_list(charges)

    assert_allclose(am1bcc_ref_charges[ref_key], charges, rtol=1e-4)


def test_dry_many_molecules_solvent(benzene_many_solv_system, tmp_path):
    """
    A basic test flushing "will it work if you pass multiple molecules"
    """
    settings = PlainMDProtocol.default_settings()
    settings.engine_settings.compute_platform = None

    protocol = PlainMDProtocol(settings=settings)

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_many_solv_system,
        stateB=benzene_many_solv_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    dag_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)["debug"]["system"]


BENZ = """\
benzene
  PyMOL2.5          3D                             0

 12 12  0  0  0  0  0  0  0  0999 V2000
    1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7022    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5079   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2540    2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2540    2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5079   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2540   -2.1719    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2540   -2.1720    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  1  6  1  0  0  0  0
  1  7  1  0  0  0  0
  2  3  1  0  0  0  0
  2  8  1  0  0  0  0
  3  4  2  0  0  0  0
  3  9  1  0  0  0  0
  4  5  1  0  0  0  0
  4 10  1  0  0  0  0
  5  6  2  0  0  0  0
  5 11  1  0  0  0  0
  6 12  1  0  0  0  0
M  END
$$$$
"""


PYRIDINE = """\
pyridine
  PyMOL2.5          3D                             0

 11 11  0  0  0  0  0  0  0  0999 V2000
    1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023    1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4045   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7023   -1.2164    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4940   -0.0325    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2473   -2.1604    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2473   -2.1604    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4945   -0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2753    2.1437    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7525    1.3034    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  1 11  2  0  0  0  0
  2  3  2  0  0  0  0
  2 10  1  0  0  0  0
  3  4  1  0  0  0  0
  3  9  1  0  0  0  0
  4  5  2  0  0  0  0
  4  8  1  0  0  0  0
  5  7  1  0  0  0  0
  2 11  1  0  0  0  0
M  END
$$$$
"""


def test_dry_run_ligand_tip4p(benzene_system, tmp_path):
    """
    Test that we can create a system with virtual sites in the
    environment (waters)
    """
    settings = PlainMDProtocol.default_settings()
    settings.engine_settings.compute_platform = None
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",  # ff14SB protein force field
        "amber/tip4pew_standard.xml",  # FF we are testing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    # we need a larger padding distance when using the dodecahedron box
    settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer
    settings.forcefield_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.solvation_settings.solvent_model = "tip4pew"
    settings.integrator_settings.reassign_velocities = True

    protocol = PlainMDProtocol(settings=settings)
    dag = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    result = dag_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)
    system = result["debug"]["system"]
    assert system


@pytest.mark.slow
def test_dry_run_complex(benzene_complex_system, tmp_path):
    # this will be very time consuming
    settings = PlainMDProtocol.default_settings()
    settings.engine_settings.compute_platform = None

    protocol = PlainMDProtocol(settings=settings)
    dag = protocol.create(
        stateA=benzene_complex_system,
        stateB=benzene_complex_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    result = dag_unit.run(dry=True, scratch_basepath=tmp_path, shared_basepath=tmp_path)
    sim = result["debug"]["system"]
    assert ThermodynamicState(
        sim,
        temperature=to_openmm(protocol.settings.thermo_settings.temperature),
    ).is_periodic
    assert isinstance(
        ThermodynamicState(
            sim,
            temperature=to_openmm(protocol.settings.thermo_settings.temperature),
        ).barostat,
        MonteCarloBarostat,
    )
    assert (
        ThermodynamicState(
            sim,
            temperature=to_openmm(protocol.settings.thermo_settings.temperature),
        ).pressure
        == 1 * omm_unit.bar
    )


def test_hightimestep(benzene_vacuum_system, tmp_path):
    settings = PlainMDProtocol.default_settings()
    settings.forcefield_settings.hydrogen_mass = 1.0
    settings.forcefield_settings.nonbonded_method = "nocutoff"

    p = PlainMDProtocol(settings=settings)
    errmsg = "too large for hydrogen mass"
    # make sure this is triggered in validate
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=benzene_vacuum_system,
            mapping=None,
        )


def test_vaccuum_PME_error(benzene_vacuum_system):
    p = PlainMDProtocol(
        settings=PlainMDProtocol.default_settings(),
    )
    errmsg = "PME cannot be used for vacuum transform"
    # make sure this is triggered in validate
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=benzene_vacuum_system,
            mapping=None,
        )


def test_multiple_basesolvents_error(a2a_protein_membrane_component):
    p = PlainMDProtocol(
        settings=PlainMDProtocol.default_settings(),
    )
    system = ChemicalSystem(
        {
            "protein-membrane": a2a_protein_membrane_component,
            "solvent": openfe.SolventComponent(),
        }
    )
    errmsg = "Multiple BaseSolventComponents found, only one is supported."
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=system,
            stateB=system,
            mapping=None,
        )


def test_states_not_matching_error(benzene_vacuum_system, toluene_vacuum_system):
    p = PlainMDProtocol(settings=PlainMDProtocol.default_settings())
    errmsg = "The two end states do not match."
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            mapping=None,
        )


def test_mapping_warning(benzene_vacuum_system, tmp_path):
    settings = PlainMDProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    p = PlainMDProtocol(settings=settings)
    warnmsg = "A mapping was passed but is not used by this Protocol."
    benzene = benzene_vacuum_system.components["ligand"]
    with pytest.warns(match=warnmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=benzene_vacuum_system,
            mapping=LigandAtomMapping(
                componentA=benzene,
                componentB=benzene,
                componentA_to_componentB=dict((i, i) for i in range(12)),
            ),
        )


@pytest.fixture
def solvent_protocol_dag(benzene_system):
    settings = PlainMDProtocol.default_settings()
    settings.protocol_repeats = 3
    protocol = PlainMDProtocol(settings=settings)

    return protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )


def test_unit_tagging(benzene_system, tmp_path):
    # test that executing the Units includes correct generation and repeat info
    settings = PlainMDProtocol.default_settings()
    settings.protocol_repeats = 3
    protocol = PlainMDProtocol(settings=settings)
    dag = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag_units = dag.protocol_units

    with mock.patch(
        "openfe.protocols.openmm_md.plain_md_methods.PlainMDSimulationUnit.run",
        return_value={
            "nc": "simulation.xtc",
            "last_checkpoint": "checkpoint.chk",
        },
    ):
        results = []
        for u in dag_units:
            # just execute the setup unit so we don't have to pass the results though to the simulation unit
            if isinstance(u, PlainMDSetupUnit):
                ret = u.execute(context=gufe.Context(tmp_path, tmp_path))
                results.append(ret)

    repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs["generation"] == 0
        repeats.add(ret.outputs["repeat_id"])
    # repeats are random ints, so check we got 3 individual numbers
    assert len(repeats) == 3


def test_gather(solvent_protocol_dag, tmp_path):
    # check .gather behaves as expected
    with mock.patch(
        "openfe.protocols.openmm_md.plain_md_methods.PlainMDSimulationUnit.run",
        return_value={
            "nc": "simulation.xtc",
            "last_checkpoint": "checkpoint.chk",
        },
    ):
        dagres = gufe.protocols.execute_DAG(
            solvent_protocol_dag,
            shared_basedir=tmp_path,
            scratch_basedir=tmp_path,
            keep_shared=True,
        )

    settings = PlainMDProtocol.default_settings()
    settings.protocol_repeats = 3
    prot = PlainMDProtocol(settings=settings)

    res = prot.gather([dagres])

    assert isinstance(res, PlainMDProtocolResult)


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, md_json):
        d = json.loads(md_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, md_json):
        d = json.loads(md_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openmm_md.plain_md_methods.PlainMDProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est is None

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est is None

    def test_get_traj_filename(self, protocolresult):
        traj = protocolresult.get_traj_filename()

        assert isinstance(traj, list)
        assert isinstance(traj[0], pathlib.Path)

    def test_get_pdb_filename(self, protocolresult):
        pdb = protocolresult.get_pdb_filename()

        assert isinstance(pdb, list)
        assert isinstance(pdb[0], pathlib.Path)
