import pathlib
from importlib import resources

import pytest
from click.testing import CliRunner

from openfecli.commands.build_warehouse import build_warehouse

from ..utils import assert_click_success


@pytest.fixture
def alchemical_network_mcl1_path() -> pathlib.Path:
    with resources.path(
        "openfe.tests.data.warehouse", "alchemicalNetwork_mc1_small.json"
    ) as fspath:
        return fspath


def test_build_warehouse(alchemical_network_mcl1_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            build_warehouse, ["--alchemical-network", str(alchemical_network_mcl1_path)]
        )
        assert_click_success(result)
        assert pathlib.Path("alchemicalNetwork_mc1_small").is_dir()
        assert pathlib.Path("alchemicalNetwork_mc1_small.db").is_file()
