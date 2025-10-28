import glob
from typing import Callable
from click.testing import CliRunner
import os
import pathlib
import pytest
import pooch

from ..utils import assert_click_success
from ..conftest import HAS_INTERNET

from openfecli.commands.gather import (
    gather,
)

POOCH_CACHE = pooch.os_cache("openfe")
ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.17435569",
    registry={"septop_results.zip": "md5:2cfa18da59a20228f5c75a1de6ec879e"},
    retry_if_failed=2,
)


@pytest.fixture
def septop_result_dir() -> pathlib.Path:
    ZENODO_RBFE_DATA.fetch(f"septop_results.zip", processor=pooch.Unzip())
    result_dir = pathlib.Path(POOCH_CACHE) / f"septop_results.untar/septop_results/"

    return result_dir


def test_septop_gather(septop_result_dir, dataset):
    results = septop_result_dir(dataset)

class TestGatherSepTop:
    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_septop_full_results(self, septop_result_dir, report, file_regression):
        results = [str(septop_result_dir / f"results_{i}") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_septop_missing_edge(self, septop_result_dir, report, file_regression):
        results = [str(septop_result_dir / f"results_{i}_remove_edge") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])
        file_regression.check(cli_result.stdout, extension=".tsv")

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["ddg", "raw"])
    def test_septop_failed_edge(self, septop_result_dir, report, file_regression):
        results = [str(septop_result_dir / f"results_{i}_failed_edge") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")
