from pathlib import Path

import pooch
import pytest
from click.testing import CliRunner

from openfecli.commands.setup_task_campaign import setup_task_campaign

from ..utils import assert_click_success


# redefined b/c cannot share fixtures between openfe and openfecli
@pytest.fixture
def alchemical_network_mcl1_path(warehouse_data_cache_dir) -> Path:
    return warehouse_data_cache_dir / "alchemicalNetwork_mcl1_small.json"


def test_setup_task_campaign(alchemical_network_mcl1_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            setup_task_campaign, ["--alchemical-network", str(alchemical_network_mcl1_path)]
        )
        assert_click_success(result)
        assert Path("warehouse_alchemicalNetwork_mcl1_small").is_dir()
        assert Path("tasks_alchemicalNetwork_mcl1_small.db").is_file()


def test_setup_task_campaign_custom_name(alchemical_network_mcl1_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            setup_task_campaign,
            [
                "--alchemical-network",
                str(alchemical_network_mcl1_path),
                "--name",
                "my_special_campaign",
            ],
        )
        assert_click_success(result)
        assert Path("warehouse_my_special_campaign").is_dir()
        assert Path("tasks_my_special_campaign.db").is_file()
