# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip
import itertools
import json
import sys
from math import sqrt
from unittest import mock

import gufe
import mdtraj as mdt
import numpy as np
import openfe
import pytest
from numpy.testing import assert_allclose
from openfe import ChemicalSystem, SolventComponent
from openfe.protocols import openmm_afe
from openfe.protocols.openmm_afe import (
    AbsoluteBindingComplexUnit,
    AbsoluteBindingProtocol,
    AbsoluteBindingSolventUnit,
)
from openfe.protocols.openmm_utils import system_validation
from openff.units import unit as offunit
from openff.units.openmm import ensure_quantity, from_openmm
from openmm import CustomNonbondedForce, NonbondedForce
from openmmtools.multistate.multistatesampler import MultiStateSampler


@pytest.fixture()
def default_settings():
    return AbsoluteBindingProtocol.default_settings()


def test_create_default_protocol(default_settings):
    # this is roughly how it should be created
    protocol = AbsoluteBindingProtocol(
        settings=default_settings,
    )
    assert protocol


def test_serialize_protocol(default_settings):
    protocol = AbsoluteBindingProtocol(
        settings=default_settings,
    )

    ser = protocol.to_dict()
    ret = AbsoluteBindingProtocol.from_dict(ser)
    assert protocol == ret


def test_unit_tagging(benzene_complex_dag, tmpdir):
    # test that executing the units includes correct gen and repeat info

    dag_units = benzene_complex_dag.protocol_units

    with (
        mock.patch('openfe.protocols.openmm_afe.equil_binding_afe_method.AbsoluteBindingSolventUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
        mock.patch('openfe.protocols.openmm_afe.equil_binding_afe_method.AbsoluteBindingComplexUnit.run',
                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
    ):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    solv_repeats = set()
    complex_repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs['generation'] == 0
        if ret.outputs['simtype'] == 'complex':
            complex_repeats.add(ret.outputs['repeat_id'])
        else:
            solv_repeats.add(ret.outputs['repeat_id'])
    # Repeat ids are random ints so just check their lengths
    assert len(complex_repeats) == len(solv_repeats) == 3


def test_create_independent_repeat_ids(benzene_modifications, T4_protein_component):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()

    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=s,
    )

    stateA = gufe.ChemicalSystem(
        {
            "protein": T4_protein_component,
            "benzene": benzene_modifications["benzene"],
            "solvent": gufe.SolventComponent(),
        }
    )

    stateB = gufe.ChemicalSystem(
        {
            "protein": T4_protein_component,
            "solvent": gufe.SolventComponent(),
        }
    )

    dags = []
    for i in range(2):
        dags.append(protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None
        ))

    repeat_ids = set()

    for dag in dags:
        for u in dag.protocol_units:
            repeat_ids.add(u.inputs['repeat_id'])

    assert len(repeat_ids) == 12


class BaseABFESystemTests:
    @pytest.fixture(scope="class")
    def protocol(self, settings):
        return openmm_afe.AbsoluteBindingProtocol(
            settings=s,
        )

    @pytest.fixture(scope="class")
    def dag(self, protocol, stateA, stateB):
        return protocol.create(
            stateA=stateA,
            stateB=stateB,
            mapping=None,
        )

    @pytest.fixture(scope="class")
    def complex_units(self, dag):
        return [u for u in prot_units if isinstance(u, AbsoluteBindingComplexUnit)]

    @pytest.fixture(scope="class")
    def solvent_units(self, dag):
        return [u for u in prot_units if isinstance(u, AbsoluteBindingSolventUnit)]


class BaseBenzeneT4Tests(BaseABFESystemTests):
    @pytest.fixture(scope="class")
    def solvent(self):
        return SolventComponent(ion_concentration=0 * offunit.molar)

    @pytest.fixture(scope="class")
    def settings(self):
        s = openmm_afe.AbsoluteBindingProtocol.default_settings()
        s.protocol_repeats = 1
        s.complex_output_settings.output_indices = "not water"
        return s

    @pytest.fixture(scope="class")
    def stateA(self, benzene_modifications, T4_protein_component, solvent):
        return ChemicalSystem(
            {
                "benzene": benzene_modifications["benzene"],
                "protein": T4_protein_component,
                "solvent": solvent,
            }
        )

    @pytest.fixture(scope="class")
    def stateB(self, T4_protein_component, solvent):
        return ChemicalSystem(
            {
                "protein": T4_protein_component,
                "solvent": solvent,
            }
        )

    @pytest.fixture(scope="class")
    def sampler(self, request):
        phase_unit = request.getfixturevalue(self.phase_unit_name)
        with tmpdir.as_cwd():
            return phase_unit[0].run(dry=True, verbose=True)["debug"]["sampler"]

    def test_number_of_units(self, dag, complex_units, solvent_units):
        assert len(list(dag.protocol_units)) == 2
        assert len(complex_units) == 1
        assert len(solvent_units) == 1

    def test_sampler_periodicity(self, sampler):
        assert sampler.is_periodic


# class TestBenzeneSolventDry(BaseBenzeneT4Tests):
#    @pytest.fixture(scope='class')
#    def sampler(self, solvent_units


# def test_dry_run_solvent_benzene(
#    benzene_modifications, T4_protein_component, tmpdir,
# ):
#
#    with tmpdir.as_cwd():
#
#        pdb = mdt.load_pdb('alchemical_system.pdb')
#        assert pdb.n_atoms == 12


def test_dry_run_complex_benzene(benzene_modifications, T4_protein_component, tmpdir):
    s = openmm_afe.AbsoluteBindingProtocol.default_settings()
    s.protocol_repeats = 1
    s.complex_output_settings.output_indices = "not water"

    protocol = openmm_afe.AbsoluteBindingProtocol(
        settings=s,
    )

    stateA = ChemicalSystem(
        {
            "benzene": benzene_modifications["benzene"],
            "protein": T4_protein_component,
            "solvent": SolventComponent(),
        }
    )

    stateB = ChemicalSystem(
        {
            "protein": T4_protein_component,
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

    assert len(prot_units) == 2

    comp_unit = [u for u in prot_units if isinstance(u, AbsoluteBindingComplexUnit)]
    sol_unit = [u for u in prot_units if isinstance(u, AbsoluteBindingSolventUnit)]

    assert len(comp_unit) == 1
    assert len(sol_unit) == 1

    with tmpdir.as_cwd():
        comp_sampler = comp_unit[0].run(dry=True, verbose=True)["debug"]["sampler"]
        assert comp_sampler.is_periodic

        pdb = mdt.load_pdb("alchemical_system.pdb")
        assert pdb.n_atoms == 2698


# def test_dry_run_solv_user_charges_benzene(benzene_modifications, tmpdir):
#    """
#    Create a test system with fictitious user supplied charges and
#    ensure that they are properly passed through to the constructed
#    alchemical system.
#    """
#    s = openmm_afe.AbsoluteSolvationProtocol.default_settings()
#    s.protocol_repeats = 1
#
#    protocol = openmm_afe.AbsoluteSolvationProtocol(
#            settings=s,
#    )
#
#    def assign_fictitious_charges(offmol):
#        """
#        Get a random array of fake partial charges for your offmol.
#        """
#        rand_arr = np.random.randint(1, 10, size=offmol.n_atoms) / 100
#        rand_arr[-1] = -sum(rand_arr[:-1])
#        return rand_arr * offunit.elementary_charge
#
#    benzene_offmol = benzene_modifications['benzene'].to_openff()
#    offmol_pchgs = assign_fictitious_charges(benzene_offmol)
#    benzene_offmol.partial_charges = offmol_pchgs
#    benzene_smc = openfe.SmallMoleculeComponent.from_openff(benzene_offmol)
#
#    # check propchgs
#    prop_chgs = benzene_smc.to_dict()['molprops']['atom.dprop.PartialCharge']
#    prop_chgs = np.array(prop_chgs.split(), dtype=float)
#    np.testing.assert_allclose(prop_chgs, offmol_pchgs)
#
#    # Create ChemicalSystems
#    stateA = ChemicalSystem({
#        'benzene': benzene_smc,
#        'solvent': SolventComponent()
#    })
#
#    stateB = ChemicalSystem({
#        'solvent': SolventComponent(),
#    })
#
#    # Create DAG from protocol, get the vacuum and solvent units
#    # and eventually dry run the first solvent unit
#    dag = protocol.create(stateA=stateA, stateB=stateB, mapping=None,)
#    prot_units = list(dag.protocol_units)
#
#    vac_unit = [u for u in prot_units
#                if isinstance(u, AbsoluteSolvationVacuumUnit)][0]
#    sol_unit = [u for u in prot_units
#                if isinstance(u, AbsoluteSolvationSolventUnit)][0]
#
#    # check sol_unit charges
#    with tmpdir.as_cwd():
#        sampler = sol_unit.run(dry=True)['debug']['sampler']
#        system = sampler._thermodynamic_states[0].system
#        nonbond = [f for f in system.getForces()
#                   if isinstance(f, NonbondedForce)]
#
#        assert len(nonbond) == 1
#
#        # loop through the 12 benzene atoms
#        # partial charge is stored in the offset
#        for i in range(12):
#            offsets = nonbond[0].getParticleParameterOffset(i)
#            c = ensure_quantity(offsets[2], 'openff')
#            assert pytest.approx(c) == prop_chgs[i]
#
#    # check vac_unit charges
#    with tmpdir.as_cwd():
#        sampler = vac_unit.run(dry=True)['debug']['sampler']
#        system = sampler._thermodynamic_states[0].system
#        nonbond = [f for f in system.getForces()
#                   if isinstance(f, CustomNonbondedForce)]
#        assert len(nonbond) == 4
#
#        custom_elec = [
#            n for n in nonbond if
#            n.getGlobalParameterName(0) == 'lambda_electrostatics'][0]
#
#        # loop through the 12 benzene atoms
#        for i in range(12):
#            c, s = custom_elec.getParticleParameters(i)
#            c = ensure_quantity(c, 'openff')
#            assert pytest.approx(c) == prop_chgs[i]


# def test_unit_tagging(benzene_solvation_dag, tmpdir):
#    # test that executing the units includes correct gen and repeat info
#
#    dag_units = benzene_solvation_dag.protocol_units
#
#    with (
#        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationSolventUnit.run',
#                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
#        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationVacuumUnit.run',
#                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
#    ):
#        results = []
#        for u in dag_units:
#            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
#            results.append(ret)
#
#    solv_repeats = set()
#    vac_repeats = set()
#    for ret in results:
#        assert isinstance(ret, gufe.ProtocolUnitResult)
#        assert ret.outputs['generation'] == 0
#        if ret.outputs['simtype'] == 'vacuum':
#            vac_repeats.add(ret.outputs['repeat_id'])
#        else:
#            solv_repeats.add(ret.outputs['repeat_id'])
#    # Repeat ids are random ints so just check their lengths
#    assert len(vac_repeats) == len(solv_repeats) == 3


# def test_gather(benzene_solvation_dag, tmpdir):
#    # check that .gather behaves as expected
#    with (
#        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationSolventUnit.run',
#                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
#        mock.patch('openfe.protocols.openmm_afe.equil_solvation_afe_method.AbsoluteSolvationVacuumUnit.run',
#                   return_value={'nc': 'file.nc', 'last_checkpoint': 'chck.nc'}),
#    ):
#        dagres = gufe.protocols.execute_DAG(benzene_solvation_dag,
#                                            shared_basedir=tmpdir,
#                                            scratch_basedir=tmpdir,
#                                            keep_shared=True)
#
#    protocol = AbsoluteSolvationProtocol(
#        settings=AbsoluteSolvationProtocol.default_settings(),
#    )
#
#    res = protocol.gather([dagres])
#
#    assert isinstance(res, openmm_afe.AbsoluteSolvationProtocolResult)
