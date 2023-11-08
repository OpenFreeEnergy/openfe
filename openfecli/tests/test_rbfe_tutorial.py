"""
Tests the easy start guide

- runs plan_rbfe_network with tyk2 inputs and checks the network created
- mocks the calculations and performs gathers on the mocked outputs
"""

import pytest
from importlib import resources
from click.testing import CliRunner
from os import path
from unittest import mock
from openff.units import unit

from openfecli.commands.plan_rbfe_network import plan_rbfe_network
from openfecli.commands.quickrun import quickrun
from openfecli.commands.gather import gather


@pytest.fixture
def tyk2_ligands():
    with resources.files('openfecli.tests.data.rbfe_tutorial') as d:
        yield str(d / 'tyk2_ligands.sdf')


@pytest.fixture
def tyk2_protein():
    with resources.files('openfecli.tests.data.rbfe_tutorial') as d:
        yield str(d / 'tyk2_protein.pdb')


@pytest.fixture
def expected_transformations():
    return ['easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json',
            'easy_rbfe_lig_ejm_31_solvent_lig_ejm_50_solvent.json',
            'easy_rbfe_lig_ejm_31_complex_lig_ejm_46_complex.json',
            'easy_rbfe_lig_ejm_42_complex_lig_ejm_43_complex.json',
            'easy_rbfe_lig_ejm_31_complex_lig_ejm_47_complex.json',
            'easy_rbfe_lig_ejm_42_solvent_lig_ejm_43_solvent.json',
            'easy_rbfe_lig_ejm_31_complex_lig_ejm_48_complex.json',
            'easy_rbfe_lig_ejm_46_complex_lig_jmc_23_complex.json',
            'easy_rbfe_lig_ejm_31_complex_lig_ejm_50_complex.json',
            'easy_rbfe_lig_ejm_46_complex_lig_jmc_27_complex.json',
            'easy_rbfe_lig_ejm_31_solvent_lig_ejm_42_solvent.json',
            'easy_rbfe_lig_ejm_46_complex_lig_jmc_28_complex.json',
            'easy_rbfe_lig_ejm_31_solvent_lig_ejm_46_solvent.json',
            'easy_rbfe_lig_ejm_46_solvent_lig_jmc_23_solvent.json',
            'easy_rbfe_lig_ejm_31_solvent_lig_ejm_47_solvent.json',
            'easy_rbfe_lig_ejm_46_solvent_lig_jmc_27_solvent.json',
            'easy_rbfe_lig_ejm_31_solvent_lig_ejm_48_solvent.json',
            'easy_rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json']


def test_plan_tyk2(tyk2_ligands, tyk2_protein, expected_transformations):
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, ['-M', tyk2_ligands,
                                                   '-p', tyk2_protein])

        assert result.exit_code == 0

        assert path.exists('alchemicalNetwork/transformations')
        for f in expected_transformations:
            assert path.exists(
                path.join('alchemicalNetwork/transformations', f))


@pytest.fixture
def mock_execute(expected_transformations):
    def fake_execute(*args, **kwargs):
        return {
            'repeat_id': kwargs['repeat_id'],
            'generation': kwargs['generation'],
            'nc': 'file.nc',
            'last_checkpoint': 'checkpoint.nc',
            'unit_estimate': 4.2 * unit.kilocalories_per_mole
        }

    with mock.patch('openfe.protocols.openmm_rfe.equil_rfe_methods.'
                    'RelativeHybridTopologyProtocolUnit._execute') as m:
        m.side_effect = fake_execute

        yield m


@pytest.fixture
def ref_gather():
    return """\
leg	ligand_i	ligand_j	DG(i->j) (kcal/mol)	uncertainty (kcal/mol)
complex	lig_ejm_31	lig_ejm_42	4.2	0.0
solvent	lig_ejm_31	lig_ejm_42	4.2	0.0
complex	lig_ejm_31	lig_ejm_46	4.2	0.0
solvent	lig_ejm_31	lig_ejm_46	4.2	0.0
complex	lig_ejm_31	lig_ejm_47	4.2	0.0
solvent	lig_ejm_31	lig_ejm_47	4.2	0.0
complex	lig_ejm_31	lig_ejm_48	4.2	0.0
solvent	lig_ejm_31	lig_ejm_48	4.2	0.0
complex	lig_ejm_31	lig_ejm_50	4.2	0.0
solvent	lig_ejm_31	lig_ejm_50	4.2	0.0
complex	lig_ejm_42	lig_ejm_43	4.2	0.0
solvent	lig_ejm_42	lig_ejm_43	4.2	0.0
complex	lig_ejm_46	lig_jmc_23	4.2	0.0
solvent	lig_ejm_46	lig_jmc_23	4.2	0.0
complex	lig_ejm_46	lig_jmc_27	4.2	0.0
solvent	lig_ejm_46	lig_jmc_27	4.2	0.0
complex	lig_ejm_46	lig_jmc_28	4.2	0.0
solvent	lig_ejm_46	lig_jmc_28	4.2	0.0
"""


def test_run_tyk2(tyk2_ligands, tyk2_protein, expected_transformations,
                  mock_execute, ref_gather):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(plan_rbfe_network, ['-M', tyk2_ligands,
                                                   '-p', tyk2_protein])

        assert result.exit_code == 0

        for f in expected_transformations:
            fn = path.join('alchemicalNetwork/transformations', f)
            result2 = runner.invoke(quickrun, [fn])
            assert result2.exit_code == 0

        gather_result = runner.invoke(gather, ["--report", "dg-raw", '.'])

        assert gather_result.exit_code == 0
        assert gather_result.stdout == ref_gather
