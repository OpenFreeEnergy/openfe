import glob
from click.testing import CliRunner
import os
import pathlib
import pytest
import pooch

from ..utils import assert_click_success
from ..conftest import HAS_INTERNET

from unittest import mock
from openfecli.commands.gather import (
    gather,
    format_estimate_uncertainty,
    _get_column,
    _load_valid_result_json,
    _get_legs_from_result_jsons,
)
from openfecli.commands.gather_septop import gather_septop

POOCH_CACHE = pooch.os_cache("openfe")
ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15042470",
    registry={
        "rbfe_results_serial_repeats.tar.gz": "md5:2355ecc80e03242a4c7fcbf20cb45487",
        "rbfe_results_parallel_repeats.tar.gz": "md5:ff7313e14eb6f2940c6ffd50f2192181",
    },
    retry_if_failed=5,
)
ZENODO_CMET_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15200083",
    registry={"cmet_results.tar.gz": "md5:a4ca67a907f744c696b09660dc1eb8ec"},
    retry_if_failed=5,
)


@pytest.mark.parametrize(
    "est,unc,unc_prec,est_str,unc_str",
    [
        (12.432, 0.111, 2, "12.43", "0.11"),
        (0.9999, 0.01, 2, "1.000", "0.010"),
        (1234, 100, 2, "1230", "100"),
    ],
)
def test_format_estimate_uncertainty(est, unc, unc_prec, est_str, unc_str):
    assert format_estimate_uncertainty(est, unc, unc_prec) == (est_str, unc_str)


@pytest.mark.parametrize(
    "val, col",
    [
        (1.0, 1),
        (0.1, -1),
        (-0.0, 0),
        (0.0, 0),
        (0.2, -1),
        (0.9, -1),
        (0.011, -2),
        (9, 1),
        (10, 2),
        (15, 2),
    ],
)
def test_get_column(val, col):
    assert _get_column(val) == col


class TestResultLoading:
    @pytest.fixture
    def sim_result(self):
        result = {
            "estimate": {},
            "uncertainty": {},
            "protocol_result": {
                "data": {
                    "22940961": [
                        {
                            "name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0",
                            "inputs": {"stateA": {"components": {"ligand": None, "solvent": None}}},
                        }
                    ]
                }
            },
            "unit_results": {
                "ProtocolUnitResult-e85": {
                    "name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0"
                },
                "ProtocolUnitFailure-4c9": {
                    "name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0",
                    "exception": ["Simulation_NanError"],
                },
            },
        }
        yield result

    def test_minimal_valid_results(self, capsys, sim_result):
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), sim_result)
            assert captured.err == ""

    def test_skip_missing_unit_result(self, capsys, sim_result):
        del sim_result["unit_results"]

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (None, None)
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err

    def test_skip_missing_estimate(self, capsys, sim_result):
        sim_result["estimate"] = None

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "No 'estimate' found" in captured.err

    def test_skip_missing_uncertainty(self, capsys, sim_result):
        sim_result["uncertainty"] = None

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "No 'uncertainty' found" in captured.err

    def test_skip_all_failed_runs(self, capsys, sim_result):
        del sim_result["unit_results"]["ProtocolUnitResult-e85"]
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == ((("lig_ejm_31", "lig_ejm_42"), "solvent"), None)
            assert "Exception found in all" in captured.err

    def test_missing_pr_data(self, capsys, sim_result):
        sim_result["protocol_result"]["data"] = {}
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (None, None)
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err

    def test_get_legs_from_result_jsons(self, capsys, sim_result):
        """Test that exceptions are handled correctly at the _get_legs_from_results_json level."""
        sim_result["protocol_result"]["data"] = {}

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = _get_legs_from_result_jsons(result_fns=[""], report="dg")
            captured = capsys.readouterr()
            assert result == {}
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err


def test_no_results_found():
    runner = CliRunner()
    cli_result = runner.invoke(gather, "not_a_file.txt")
    assert cli_result.exit_code == 1
    assert "No results JSON files found" in str(cli_result.stderr)


_RBFE_EXPECTED_DG = b"""
ligand	DG(MLE) (kcal/mol)	uncertainty (kcal/mol)
lig_ejm_31	-0.09	0.05
lig_ejm_42	0.7	0.1
lig_ejm_46	-0.98	0.05
lig_ejm_47	-0.1	0.1
lig_ejm_48	0.53	0.09
lig_ejm_50	0.91	0.06
lig_ejm_43	2.0	0.2
lig_jmc_23	-0.68	0.09
lig_jmc_27	-1.1	0.1
lig_jmc_28	-1.25	0.08
"""

_RBFE_EXPECTED_DDG = b"""
ligand_i	ligand_j	DDG(i->j) (kcal/mol)	uncertainty (kcal/mol)
lig_ejm_31	lig_ejm_42	0.8	0.1
lig_ejm_31	lig_ejm_46	-0.89	0.06
lig_ejm_31	lig_ejm_47	0.0	0.1
lig_ejm_31	lig_ejm_48	0.61	0.09
lig_ejm_31	lig_ejm_50	1.00	0.04
lig_ejm_42	lig_ejm_43	1.4	0.2
lig_ejm_46	lig_jmc_23	0.29	0.09
lig_ejm_46	lig_jmc_27	-0.1	0.1
lig_ejm_46	lig_jmc_28	-0.27	0.06
"""

_RBFE_EXPECTED_RAW = b"""\
leg	ligand_i	ligand_j	DG(i->j) (kcal/mol)	MBAR uncertainty (kcal/mol)
complex	lig_ejm_31	lig_ejm_42	-14.9	0.8
complex	lig_ejm_31	lig_ejm_42	-14.8	0.8
complex	lig_ejm_31	lig_ejm_42	-15.1	0.8
solvent	lig_ejm_31	lig_ejm_42	-15.7	0.8
solvent	lig_ejm_31	lig_ejm_42	-15.7	0.8
solvent	lig_ejm_31	lig_ejm_42	-15.7	0.8
complex	lig_ejm_31	lig_ejm_46	-40.7	0.8
complex	lig_ejm_31	lig_ejm_46	-40.7	0.8
complex	lig_ejm_31	lig_ejm_46	-40.8	0.8
solvent	lig_ejm_31	lig_ejm_46	-39.8	0.8
solvent	lig_ejm_31	lig_ejm_46	-39.9	0.8
solvent	lig_ejm_31	lig_ejm_46	-39.8	0.8
complex	lig_ejm_31	lig_ejm_47	-27.8	0.8
complex	lig_ejm_31	lig_ejm_47	-28.0	0.8
complex	lig_ejm_31	lig_ejm_47	-27.7	0.8
solvent	lig_ejm_31	lig_ejm_47	-27.8	0.8
solvent	lig_ejm_31	lig_ejm_47	-27.8	0.8
solvent	lig_ejm_31	lig_ejm_47	-27.9	0.8
complex	lig_ejm_31	lig_ejm_48	-16.2	0.8
complex	lig_ejm_31	lig_ejm_48	-16.2	0.8
complex	lig_ejm_31	lig_ejm_48	-16.0	0.8
solvent	lig_ejm_31	lig_ejm_48	-16.8	0.8
solvent	lig_ejm_31	lig_ejm_48	-16.7	0.8
solvent	lig_ejm_31	lig_ejm_48	-16.8	0.8
complex	lig_ejm_31	lig_ejm_50	-57.3	0.8
complex	lig_ejm_31	lig_ejm_50	-57.3	0.8
complex	lig_ejm_31	lig_ejm_50	-57.4	0.8
solvent	lig_ejm_31	lig_ejm_50	-58.3	0.8
solvent	lig_ejm_31	lig_ejm_50	-58.4	0.8
solvent	lig_ejm_31	lig_ejm_50	-58.3	0.8
complex	lig_ejm_42	lig_ejm_43	-19.0	0.8
complex	lig_ejm_42	lig_ejm_43	-18.7	0.8
complex	lig_ejm_42	lig_ejm_43	-19.0	0.8
solvent	lig_ejm_42	lig_ejm_43	-20.3	0.8
solvent	lig_ejm_42	lig_ejm_43	-20.3	0.8
solvent	lig_ejm_42	lig_ejm_43	-20.3	0.8
complex	lig_ejm_46	lig_jmc_23	17.3	0.8
complex	lig_ejm_46	lig_jmc_23	17.4	0.8
complex	lig_ejm_46	lig_jmc_23	17.5	0.8
solvent	lig_ejm_46	lig_jmc_23	17.2	0.8
solvent	lig_ejm_46	lig_jmc_23	17.1	0.8
solvent	lig_ejm_46	lig_jmc_23	17.1	0.8
complex	lig_ejm_46	lig_jmc_27	15.9	0.8
complex	lig_ejm_46	lig_jmc_27	15.8	0.8
complex	lig_ejm_46	lig_jmc_27	15.7	0.8
solvent	lig_ejm_46	lig_jmc_27	16.0	0.8
solvent	lig_ejm_46	lig_jmc_27	15.9	0.8
solvent	lig_ejm_46	lig_jmc_27	15.9	0.8
complex	lig_ejm_46	lig_jmc_28	23.1	0.8
complex	lig_ejm_46	lig_jmc_28	23.2	0.8
complex	lig_ejm_46	lig_jmc_28	23.1	0.8
solvent	lig_ejm_46	lig_jmc_28	23.5	0.8
solvent	lig_ejm_46	lig_jmc_28	23.3	0.8
solvent	lig_ejm_46	lig_jmc_28	23.4	0.8
"""


@pytest.fixture
def rbfe_result_dir() -> pathlib.Path:
    def _rbfe_result_dir(dataset) -> str:
        ZENODO_RBFE_DATA.fetch(f"{dataset}.tar.gz", processor=pooch.Untar())
        cache_dir = pathlib.Path(POOCH_CACHE) / f"{dataset}.tar.gz.untar/{dataset}/"
        return cache_dir

    return _rbfe_result_dir


@pytest.fixture
def cmet_result_dir() -> pathlib.Path:
    ZENODO_CMET_DATA.fetch(f"cmet_results.tar.gz", processor=pooch.Untar())
    result_dir = pathlib.Path(POOCH_CACHE) / f"cmet_results.tar.gz.untar/cmet_results/"

    return result_dir


class TestGatherCMET:
    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_cmet_full_results(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    # TODO: add --allow-partial behavior checks
    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_cmet_missing_complex_leg(self, cmet_result_dir, report, file_regression):
        """Missing one complex replicate from one leg."""
        results = [
            str(cmet_result_dir / d) for d in ["results_0_partial", "results_1", "results_2"]
        ]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_cmet_missing_edge(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}_remove_edge") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])
        file_regression.check(cli_result.stdout, extension=".tsv")

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["ddg", "raw"])
    def test_cmet_failed_edge(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}_failed_edge") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("allow_partial", [True, False])
    def test_cmet_too_few_edges_error(self, cmet_result_dir, allow_partial):
        results = [str(cmet_result_dir / f"results_{i}_failed_edge") for i in range(3)]
        args = ["--report", "dg"]
        runner = CliRunner()
        if allow_partial:
            args += ["--allow-partial"]

        cli_result = runner.invoke(gather, results + args + ["--tsv"])
        assert cli_result.exit_code == 1
        assert "The results network has 1 edge(s), but 3 or more edges are required" in str(
            cli_result.stderr
        )

    @pytest.mark.parametrize("report", ["dg", "ddg"])
    def test_cmet_missing_all_complex_legs_fail(self, cmet_result_dir, report, file_regression):
        """Missing one complex replicate from one leg."""
        results = glob.glob(f"{cmet_result_dir}/results_*/*solvent*", recursive=True)
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["-o", "-"])

        cli_result.exit_code == 1
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["ddg"])
    def test_cmet_missing_all_complex_legs_allow_partial(self, cmet_result_dir, report, file_regression):  # fmt: skip
        """Missing one complex replicate from one leg."""
        results = glob.glob(f"{cmet_result_dir}/results_*/*solvent*", recursive=True)
        args = ["--report", report, "--allow-partial"]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    def test_pretty_print(self, cmet_result_dir, report, file_regression):
        results = [str(cmet_result_dir / f"results_{i}") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather, results + args)
        assert_click_success(cli_result)
        # TODO: figure out how to mock terminal size, since it affects the table wrapping
        # file_regression.check(cli_result.stdout, extension='.txt')

    def test_write_to_file(self, cmet_result_dir):
        runner = CliRunner()
        with runner.isolated_filesystem():
            results = [str(cmet_result_dir / f"results_{i}") for i in range(3)]
            fname = "output.tsv"
            args = ["--report", "raw", "-o", fname]
            cli_result = runner.invoke(gather, results + args)
            assert "writing raw output to 'output.tsv'" in cli_result.stdout
            assert pathlib.Path(fname).is_file()


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
@pytest.mark.parametrize("dataset", ["rbfe_results_serial_repeats", "rbfe_results_parallel_repeats"])  # fmt: skip
@pytest.mark.parametrize("report", ["", "dg", "ddg", "raw"])
@pytest.mark.parametrize("input_mode", ["directory", "filepaths"])
def test_rbfe_gather(rbfe_result_dir, dataset, report, input_mode):
    expected = {
        "": _RBFE_EXPECTED_DG,
        "dg": _RBFE_EXPECTED_DG,
        "ddg": _RBFE_EXPECTED_DDG,
        "raw": _RBFE_EXPECTED_RAW,
    }[report]
    runner = CliRunner()

    if report:
        args = ["--report", report]
    else:
        args = []

    results = rbfe_result_dir(dataset)
    if input_mode == "directory":
        results = [str(results)]
    elif input_mode == "filepaths":
        results = glob.glob(f"{results}/*", recursive=True)
        assert len(results) > 1  # sanity check to make sure we're passing in multiple paths

    cli_result = runner.invoke(gather, results + args + ["--tsv"])

    assert_click_success(cli_result)

    actual_lines = set(cli_result.stdout_bytes.split(b"\n"))
    assert set(expected.split(b"\n")) == actual_lines


def test_rbfe_gather_single_repeats_dg_error(rbfe_result_dir):
    """A single repeat is insufficient for a dg calculation - should fail cleanly."""

    runner = CliRunner()
    results = rbfe_result_dir("rbfe_results_parallel_repeats")
    args = ["report", "dg"]
    cli_result = runner.invoke(gather, [f"{results}/replicate_0"] + args + ["--tsv"])
    assert cli_result.exit_code == 1


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
class TestRBFEGatherFailedEdges:
    @pytest.fixture()
    def results_paths_serial_missing_legs(self, rbfe_result_dir) -> str:
        """Example output data, with replicates run in serial and two missing results JSONs."""
        result_dir = rbfe_result_dir("rbfe_results_serial_repeats")
        results = glob.glob(f"{result_dir}/*", recursive=True)

        files_to_skip = [
            "rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json",
            "rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json",
        ]

        results_filtered = [f for f in results if os.path.basename(f) not in files_to_skip]

        return results_filtered

    def test_missing_leg_error(self, results_paths_serial_missing_legs: str):
        runner = CliRunner()
        result = runner.invoke(gather, results_paths_serial_missing_legs + ["--report", "dg"])

        assert result.exit_code == 1
        assert "Some edge(s) are missing runs" in str(result.stderr)
        assert "lig_ejm_31\tlig_ejm_42\tsolvent" in str(result.stderr)
        assert "lig_ejm_46\tlig_jmc_28\tcomplex" in str(result.stderr)
        assert "using the --allow-partial flag" in str(result.stderr)

    def test_missing_leg_allow_partial_disconnected(self, results_paths_serial_missing_legs: str):
        runner = CliRunner()
        with pytest.warns():
            args = ["--report", "dg", "--allow-partial"]
            result = runner.invoke(gather, results_paths_serial_missing_legs + args + ["--tsv"])
            assert result.exit_code == 1
            assert "The results network is disconnected" in str(result.stderr)

    def test_allow_partial_msg_not_printed(self, results_paths_serial_missing_legs: str):
        # we *dont* want the suggestion to use --allow-partial if the user already used it!
        runner = CliRunner()
        args = ["--report", "ddg", "--allow-partial"]
        result = runner.invoke(gather, results_paths_serial_missing_legs + args + ["--tsv"])
        assert_click_success(result)
        assert "--allow-partial" not in result.output


ZENODO_SEPTOP_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.17435569",
    registry={"septop_results.zip": "md5:2cfa18da59a20228f5c75a1de6ec879e"},
    retry_if_failed=2,
)


@pytest.fixture
def septop_result_dir() -> pathlib.Path:
    ZENODO_SEPTOP_DATA.fetch(f"septop_results.zip", processor=pooch.Unzip())
    result_dir = pathlib.Path(POOCH_CACHE) / f"septop_results.zip.unzip/septop_results/"

    return result_dir


def test_septop_gather(septop_result_dir, dataset):
    results = septop_result_dir(dataset)


class TestGatherSepTop:
    @pytest.mark.parametrize("report", ["raw", "ddg", "dg"])
    def test_septop_full_results(self, septop_result_dir, report, file_regression):
        results = [str(septop_result_dir / f"results_{i}") for i in range(3)]
        args = ["--report", report]
        runner = CliRunner()
        cli_result = runner.invoke(gather_septop, results + args + ["--tsv"])

        assert_click_success(cli_result)
        file_regression.check(cli_result.stdout, extension=".tsv")

    # @pytest.mark.parametrize("report", ["dg", "ddg", "raw"])
    # def test_septop_missing_edge(self, septop_result_dir, report, file_regression):
    #     results = [str(septop_result_dir / f"results_{i}_remove_edge") for i in range(3)]
    #     args = ["--report", report]
    #     runner = CliRunner()
    #     cli_result = runner.invoke(gather, results + args + ["--tsv"])
    #     file_regression.check(cli_result.stdout, extension=".tsv")

    #     assert_click_success(cli_result)
    #     file_regression.check(cli_result.stdout, extension=".tsv")

    # @pytest.mark.parametrize("report", ["ddg", "raw"])
    # def test_septop_failed_edge(self, septop_result_dir, report, file_regression):
    #     results = [str(septop_result_dir / f"results_{i}_failed_edge") for i in range(3)]
    #     args = ["--report", report]
    #     runner = CliRunner()
    #     cli_result = runner.invoke(gather, results + args + ["--tsv"])

    #     assert_click_success(cli_result)
    #     file_regression.check(cli_result.stdout, extension=".tsv")
