from unittest import mock

from click.testing import CliRunner

from openfe.tests.storage.test_warehouse import warehouse_partial_failure
from openfecli.commands.to_legacy_json import to_legacy_json

from ..utils import assert_click_success


def test_to_legacy_json(warehouse_partial_failure, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        mocked_warehouse_path = tmp_path / "warehouse"
        mocked_warehouse_path.mkdir()
        out_dir = tmp_path / "my_results"
        with mock.patch(
            "openfe.storage.warehouse.FileSystemWarehouse.from_dir",
            return_value=warehouse_partial_failure,
        ):
            result = runner.invoke(to_legacy_json, [str(mocked_warehouse_path), "-o", str(out_dir)])
        assert_click_success(result)
        assert out_dir.is_dir()
        out_files = [str(p.name) for p in out_dir.rglob("Trans*")]
        assert len(out_files) == 4


def test_to_legacy_json_empty(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = tmp_path / "warehouse"
        warehouse_path.mkdir()
        out_dir = tmp_path / "my_results"

        result = runner.invoke(to_legacy_json, [str(warehouse_path), "-o", str(out_dir)])
        assert "Loading results" in result.stdout
        assert "No results found" in result.stderr
        assert not out_dir.is_dir()
