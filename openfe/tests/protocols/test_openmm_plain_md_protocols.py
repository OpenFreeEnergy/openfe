# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
from io import StringIO
import numpy as np
import gufe
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin
import pytest
from unittest import mock
from openff.units import unit
from importlib import resources
import xml.etree.ElementTree as ET
from openff.units.openmm import from_openmm, to_openmm
from openmm import app, XmlSerializer, MonteCarloBarostat
from openmm import unit as omm_unit
from openmmtools.states import ThermodynamicState
import pathlib
from rdkit import Chem
from rdkit.Geometry import Point3D
import mdtraj as mdt

import openfe
from openfe import setup
from openfe.protocols import openmm_md
from openfe.protocols.openmm_utils import system_creation
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.units.openmm import ensure_quantity
from openfe.protocols.openmm_md.plain_md_methods import (
    PlainMDProtocol, PlainMDProtocolUnit, PlainMDProtocolResult,
)
from openfe.protocols.openmm_md.plain_md_settings import PlainMDProtocolSettings


def test_create_default_settings():
    settings = PlainMDProtocol.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = PlainMDProtocol(
        settings=PlainMDProtocol.default_settings(),
    )

    assert protocol


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
    settings.repeat_settings.n_repeats = 3
    protocol = PlainMDProtocol(
            settings=settings,
    )
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
    u: PlainMDProtocolUnit
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs['repeat_id'])

    assert len(repeat_ids) == 6


def test_dry_run_default_vacuum(benzene_vacuum_system, tmpdir):

    vac_settings = PlainMDProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'

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

    with tmpdir.as_cwd():
        sim = dag_unit.run(dry=True)['debug']['sampler']
        print(sim)
        assert not ThermodynamicState(sim, temperature=
        to_openmm(protocol.settings.thermo_settings.temperature)).is_periodic
        assert ThermodynamicState(sim, temperature=
        to_openmm(protocol.settings.thermo_settings.temperature)).barostat is None

        # Check MDTtraj Topologies
        # 16 atoms:
        # 11 common atoms, 1 extra hydrogen in benzene, 4 extra in toluene
        # 12 bonds in benzene + 4 extra tolunee bonds
        # assert len(list(.atoms)) == 16
        # assert len(list(htf.omm_hybrid_topology.atoms())) == 16
        # assert len(list(htf.hybrid_topology.bonds)) == 16
        # assert len(list(htf.omm_hybrid_topology.bonds())) == 16

        # # smoke test - can convert back the mdtraj topology
        # ret_top = mdt.Topology.to_openmm(htf.hybrid_topology)
        # assert len(list(ret_top.atoms())) == 16
        # assert len(list(ret_top.bonds())) == 16


def test_dry_run_gaff_vacuum(benzene_vacuum_system, tmpdir):
    vac_settings = PlainMDProtocol.default_settings()
    vac_settings.system_settings.nonbonded_method = 'nocutoff'
    vac_settings.forcefield_settings.small_molecule_forcefield = 'gaff-2.11'

    protocol = PlainMDProtocol(
            settings=vac_settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']


def test_dry_many_molecules_solvent(
    benzene_many_solv_system, tmpdir
):
    """
    A basic test flushing "will it work if you pass multiple molecules"
    """
    settings = PlainMDProtocol.default_settings()

    protocol = PlainMDProtocol(
            settings=settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_many_solv_system,
        stateB=benzene_many_solv_system,
        mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = unit.run(dry=True)['debug']['sampler']


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

def test_dry_run_ligand_tip4p(benzene_system, tmpdir):
    """
    Test that we can create a system with virtual sites in the
    environment (waters)
    """
    settings = PlainMDProtocol.default_settings()
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",    # ff14SB protein force field
        "amber/tip4pew_standard.xml", # FF we are testsing with the fun VS
        "amber/phosaa10.xml",  # Handles THE TPO
    ]
    settings.solvation_settings.solvent_padding = 1.0 * unit.nanometer
    settings.system_settings.nonbonded_cutoff = 0.9 * unit.nanometer
    settings.solvation_settings.solvent_model = 'tip4pew'
    settings.integrator_settings.reassign_velocities = True

    protocol = PlainMDProtocol(
            settings=settings,
    )
    dag = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        sampler = dag_unit.run(dry=True)['debug']['sampler']
        assert sampler


# @pytest.mark.slow
# def test_dry_run_complex(benzene_complex_system, tmpdir):
#     # this will be very time consuming
#     settings = PlainMDProtocol.default_settings()
#
#     protocol = PlainMDProtocol(
#             settings=settings,
#     )
#     dag = protocol.create(
#         stateA=benzene_complex_system,
#         stateB=benzene_complex_system,
#         mapping=None,
#     )
#     dag_unit = list(dag.protocol_units)[0]
#
#     with tmpdir.as_cwd():
#         sim = dag_unit.run(dry=True)['debug']['sampler']
#         assert not ThermodynamicState(sim, temperature=
#         to_openmm(protocol.settings.thermo_settings.temperature)).is_periodic
#         assert isinstance(ThermodynamicState(sim, temperature=
#         to_openmm(protocol.settings.thermo_settings.temperature)).barostat,
#                           MonteCarloBarostat)
#         assert ThermodynamicState(sim, temperature=
#         to_openmm(protocol.settings.thermo_settings.temperature)).pressure == 1 * omm_unit.bar


def test_hightimestep(benzene_vacuum_system, tmpdir):
    settings = PlainMDProtocol.default_settings()
    settings.forcefield_settings.hydrogen_mass = 1.0
    settings.system_settings.nonbonded_method = 'nocutoff'

    p = PlainMDProtocol(
            settings=settings,
    )

    dag = p.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    errmsg = "too large for hydrogen mass"
    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            dag_unit.run(dry=True)


def test_vaccuum_PME_error(benzene_vacuum_system):

    p = PlainMDProtocol(
        settings=PlainMDProtocol.default_settings(),
    )
    errmsg = "PME cannot be used for vacuum transform"
    with pytest.raises(ValueError, match=errmsg):
        _ = p.create(
            stateA=benzene_vacuum_system,
            stateB=benzene_vacuum_system,
            mapping=None,
        )


@pytest.fixture
def solvent_protocol_dag(benzene_system):
    settings = PlainMDProtocol.default_settings()
    settings.repeat_settings.n_repeats = 3
    protocol = PlainMDProtocol(
        settings=settings,
    )

    return protocol.create(
        stateA=benzene_system, stateB=benzene_system,
        mapping=None,
    )


def test_unit_tagging(solvent_protocol_dag, tmpdir):
    # test that executing the Units includes correct generation and repeat info
    dag_units = solvent_protocol_dag.protocol_units
    with mock.patch('openfe.protocols.openmm_md.plain_md_methods.PlainMDProtocolUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs['generation'] == 0
        repeats.add(ret.outputs['repeat_id'])
    # repeats are random ints, so check we got 3 individual numbers
    assert len(repeats) == 3


def test_gather(solvent_protocol_dag, tmpdir):
    # check .gather behaves as expected
    with mock.patch('openfe.protocols.openmm_md.plain_md_methods.PlainMDProtocolUnit.run',
                    return_value={'nc': 'file.nc', 'last_checkpoint': 'chk.nc'}):
        dagres = gufe.protocols.execute_DAG(solvent_protocol_dag,
                                            shared_basedir=tmpdir,
                                            scratch_basedir=tmpdir,
                                            keep_shared=True)

    settings = PlainMDProtocol.default_settings()
    settings.repeat_settings.n_repeats = 3
    prot = PlainMDProtocol(
        settings=settings
    )

    res = prot.gather([dagres])

    assert isinstance(res, PlainMDProtocolResult)
