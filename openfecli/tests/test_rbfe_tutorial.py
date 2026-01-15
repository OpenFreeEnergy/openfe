"""
Tests the easy start guide

- runs plan_rbfe_network with tyk2 inputs and checks the network created
- mocks the calculations and performs gathers on the mocked outputs
"""

import os
from importlib import resources
from os import path
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner
from openff.units import unit

from openfecli.commands.gather import gather
from openfecli.commands.plan_rbfe_network import plan_rbfe_network
from openfecli.commands.quickrun import quickrun

from .utils import assert_click_success


@pytest.fixture
def tyk2_ligands():
    with resources.as_file(resources.files("openfecli.tests.data.rbfe_tutorial")) as d:
        yield str(d / "tyk2_ligands.sdf")


@pytest.fixture
def tyk2_protein():
    with resources.as_file(resources.files("openfecli.tests.data.rbfe_tutorial")) as d:
        yield str(d / "tyk2_protein.pdb")


@pytest.fixture
def expected_transformations():
    return [
        "rbfe_lig_ejm_31_solvent_lig_ejm_48_solvent.json",
        "rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json",
        "rbfe_lig_jmc_27_complex_lig_jmc_28_complex.json",
        "rbfe_lig_jmc_23_solvent_lig_jmc_28_solvent.json",
        "rbfe_lig_ejm_42_solvent_lig_ejm_50_solvent.json",
        "rbfe_lig_ejm_31_complex_lig_ejm_46_complex.json",
        "rbfe_lig_ejm_31_solvent_lig_ejm_50_solvent.json",
        "rbfe_lig_ejm_42_solvent_lig_ejm_43_solvent.json",
        "rbfe_lig_ejm_31_complex_lig_ejm_47_complex.json",
        "rbfe_lig_jmc_27_solvent_lig_jmc_28_solvent.json",
        "rbfe_lig_jmc_23_complex_lig_jmc_28_complex.json",
        "rbfe_lig_ejm_42_complex_lig_ejm_50_complex.json",
        "rbfe_lig_ejm_31_solvent_lig_ejm_46_solvent.json",
        "rbfe_lig_ejm_31_complex_lig_ejm_50_complex.json",
        "rbfe_lig_ejm_42_complex_lig_ejm_43_complex.json",
        "rbfe_lig_ejm_31_solvent_lig_ejm_47_solvent.json",
        "rbfe_lig_ejm_31_complex_lig_ejm_48_complex.json",
        "rbfe_lig_ejm_46_complex_lig_jmc_28_complex.json",
    ]


def test_plan_tyk2(tyk2_ligands, tyk2_protein, expected_transformations):
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(
            plan_rbfe_network,
            [
                "-M", tyk2_ligands,
                "-p", tyk2_protein,
            ],
        )  # fmt: skip
        assert_click_success(result)
        assert path.exists("alchemicalNetwork/transformations")
        for f in expected_transformations:
            assert path.exists(path.join("alchemicalNetwork/transformations", f))
        # make sure these are the only transforms
        assert len(os.listdir("alchemicalNetwork/transformations")) == len(expected_transformations)

        # check that the correct default settings are used, we currently have no provenance on the object so check
        # the output string
        # check the atom mapper
        assert "Mapper: <KartografAtomMapper" in result.output
        # check the score method
        assert "Mapping Scorer: <function default_lomap_score" in result.output
        # check the network planner
        assert "Network Generation: <function generate_minimal_spanning_network" in result.output
        # check the partial charge method
        assert "Partial Charge Generation: am1bcc" in result.output
        # check the number of repeats
        assert "n_protocol_repeats=3" in result.output


@pytest.fixture
def ref_gather():
    return """\
Loading results:
ligand_i\tligand_j\tDDG(i->j) (kcal/mol)\tuncertainty (kcal/mol)
lig_ejm_31\tlig_ejm_46\t0.0\t0.0
lig_ejm_31\tlig_ejm_47\t0.0\t0.0
lig_ejm_31\tlig_ejm_48\t0.0\t0.0
lig_ejm_31\tlig_ejm_50\t0.0\t0.0
lig_ejm_42\tlig_ejm_43\t0.0\t0.0
lig_ejm_42\tlig_ejm_50\t0.0\t0.0
lig_ejm_46\tlig_jmc_28\t0.0\t0.0
lig_jmc_23\tlig_jmc_28\t0.0\t0.0
lig_jmc_27\tlig_jmc_28\t0.0\t0.0
"""


@pytest.fixture
def fake_setup_execute_results():
    """Use for mocking the expensive _execute step and instead directly return plausible results."""

    def _fake_execute_results(*args, **kwargs):
        return {
            "repeat_id": kwargs["repeat_id"],
            "generation": kwargs["generation"],
            "system": Path("system.xml.bz2"),
            "positions": Path("positions.npy"),
            "pdb_structure": Path("hybrid_system.pdb"),
            "selection_indices": np.arange(50),
        }

    return _fake_execute_results


@pytest.fixture
def fake_sim_execute_results():
    """Use for mocking the expensive _execute step and instead directly return plausible results."""

    def _fake_execute_results(*args, **kwargs):
        return {
            "repeat_id": kwargs["repeat_id"],
            "generation": kwargs["generation"],
            "nc": Path("file.nc"),
            "checkpoint": Path("chk.chk"),
        }

    return _fake_execute_results


@pytest.fixture
def fake_analysis_execute_results():
    """Use for mocking the expensive _execute step and instead directly return plausible results."""

    def _fake_execute_results(*args, **kwargs):
        return {
            "repeat_id": kwargs["repeat_id"],
            "generation": kwargs["generation"],
            "pdb_structure": Path("hybrid_system.pdb"),
            "checkpoint": Path("chk.chk"),
            "selection_indices": np.arange(50),
            "unit_estimate": 4.2 * unit.kilocalories_per_mole,
        }

    return _fake_execute_results


def test_run_tyk2(
    tyk2_ligands,
    tyk2_protein,
    expected_transformations,
    fake_setup_execute_results,
    fake_sim_execute_results,
    fake_analysis_execute_results,
    ref_gather,
):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result_setup = runner.invoke(
            plan_rbfe_network,
            [
                "-M", tyk2_ligands,
                "-p", tyk2_protein,
            ],
        )  # fmt: skip

        assert_click_success(result_setup)
        with (
            mock.patch(
                "openfe.protocols.openmm_rfe.hybridtop_units.HybridTopologySetupUnit._execute",
                side_effect=fake_setup_execute_results,
            ),
            mock.patch(
                "openfe.protocols.openmm_rfe.hybridtop_units.HybridTopologyMultiStateSimulationUnit._execute",
                side_effect=fake_sim_execute_results,
            ),
            mock.patch(
                "openfe.protocols.openmm_rfe.hybridtop_units.HybridTopologyMultiStateAnalysisUnit._execute",
                side_effect=fake_analysis_execute_results,
            ),
        ):
            for f in expected_transformations:
                fn = path.join("alchemicalNetwork/transformations", f)
                result_run = runner.invoke(quickrun, [fn])
                assert_click_success(result_run)

        result_gather = runner.invoke(gather, ["--report", "ddg", ".", "--tsv"])

        assert_click_success(result_gather)
        assert result_gather.stdout == ref_gather
