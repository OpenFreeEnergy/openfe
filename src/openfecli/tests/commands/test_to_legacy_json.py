import pathlib
from unittest import mock

from click.testing import CliRunner

from openfecli.commands.to_legacy_json import to_legacy_json

from ..utils import assert_click_success


def test_to_legacy_json(warehouse_partial_dir, tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_dir = tmp_path / "my_results"

        result = runner.invoke(to_legacy_json, [str(warehouse_partial_dir), "-o", str(out_dir)])
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


def test_to_legacy_json_dir_exists_error(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = tmp_path / "warehouse"
        warehouse_path.mkdir()
        out_dir = tmp_path / "my_results"
        out_dir.mkdir()
        result = runner.invoke(to_legacy_json, [str(warehouse_path), "-o", str(out_dir)])
        assert "cannot already exist" in str(result.stderr)


def test_to_legacy_json_dir_uses_default_out(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = tmp_path / "my_warehouse"
        warehouse_path.mkdir()
        expected_outdir = pathlib.Path("results_my_warehouse")
        expected_outdir.mkdir()
        result = runner.invoke(to_legacy_json, [str(warehouse_path)])
        assert result.exit_code == 1
        assert "cannot already exist" in str(result.stderr)
