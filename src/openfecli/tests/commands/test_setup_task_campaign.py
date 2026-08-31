import pathlib
from importlib import resources

import pytest
from click.testing import CliRunner

from openfecli.commands.setup_task_campaign import setup_task_campaign

from ..utils import assert_click_success


@pytest.fixture
def alchemical_network_mcl1_path() -> pathlib.Path:
    with resources.path(
        "openfe.tests.data.warehouse", "alchemicalNetwork_mc1_small.json"
    ) as fspath:
        return fspath


def test_setup_task_campaign(alchemical_network_mcl1_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            setup_task_campaign, ["--alchemical-network", str(alchemical_network_mcl1_path)]
        )
        assert_click_success(result)
        assert pathlib.Path("alchemicalNetwork_mc1_small").is_dir()
        assert pathlib.Path("alchemicalNetwork_mc1_small.db").is_file()
