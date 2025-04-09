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
    load_valid_result_json,
)

@pytest.mark.parametrize('est,unc,unc_prec,est_str,unc_str', [
    (12.432, 0.111, 2, "12.43", "0.11"),
    (0.9999, 0.01, 2, "1.000", "0.010"),
    (1234, 100, 2, "1230", "100"),
])
def test_format_estimate_uncertainty(est, unc, unc_prec, est_str, unc_str):
    assert format_estimate_uncertainty(est, unc, unc_prec) == (est_str, unc_str)

@pytest.mark.parametrize('val, col', [
    (1.0, 1), (0.1, -1), (-0.0, 0), (0.0, 0), (0.2, -1), (0.9, -1),
    (0.011, -2), (9, 1), (10, 2), (15, 2),
])
def test_get_column(val, col):
    assert _get_column(val) == col


class TestResultLoading:
    @pytest.fixture
    def sim_result(self):
        result = {
            "estimate": {},
            "uncertainty": {},
            "protocol_result": {'data':{'2294096155646608107660849867019691905':  [
                {"name": "lig_ejm_31 to lig_ejm_42 repeat 0 generation 0",
                  "inputs": {"stateA": {"components": {"ligand":None}}}}
                  ]}},
            "unit_results": {
                "ProtocolUnitResult-e85": {"name":"lig_ejm_31 to lig_ejm_42 repeat 0 generation 0"},
                "ProtocolUnitFailure-4c9": {"name":"lig_ejm_31 to lig_ejm_42 repeat 0 generation 0",
                                            "exception": ["Simulation_NanError"]},
            },
        }
        yield result

    def test_minimal_valid_results(self, capsys, sim_result):
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (('lig_ejm_31', 'lig_ejm_42', 'vacuum'), sim_result)
            assert captured.err == ""

    def test_skip_missing_filepath(self, capsys):
        result = load_valid_result_json(fpath="")
        captured = capsys.readouterr()
        assert result == (None, None)
        assert "does not exist. Skipping" in captured.err

    def test_skip_missing_unit_result(self, capsys, sim_result):
        del sim_result["unit_results"]

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (None, None)
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err

    def test_skip_missing_estimate(self, capsys, sim_result):
        sim_result["estimate"] = None

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (('lig_ejm_31', 'lig_ejm_42', 'vacuum'), None)
            assert "No 'estimate' found" in captured.err

    def test_skip_missing_uncertainty(self, capsys, sim_result):
        sim_result["uncertainty"] = None

        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (('lig_ejm_31', 'lig_ejm_42', 'vacuum'), None)
            assert "No 'uncertainty' found" in captured.err

    def test_skip_all_failed_runs(self, capsys, sim_result):
        del sim_result["unit_results"]["ProtocolUnitResult-e85"]
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (('lig_ejm_31', 'lig_ejm_42', 'vacuum'), None)
            assert "Exception found in all" in captured.err

    def test_missing_pr_data(self, capsys, sim_result):
        sim_result["protocol_result"]["data"] = {}
        with mock.patch("openfecli.commands.gather.load_json", return_value=sim_result):
            result = load_valid_result_json(fpath="")
            captured = capsys.readouterr()
            assert result == (None, None)
            assert "Missing ligand names and/or simulation type. Skipping" in captured.err

_EXPECTED_DG = b"""
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

_EXPECTED_DDG = b"""
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

_EXPECTED_DG_RAW = b"""
leg	ligand_i	ligand_j	DG(i->j) (kcal/mol)	uncertainty (kcal/mol)
complex	lig_ejm_31	lig_ejm_42	-15.0	0.1
solvent	lig_ejm_31	lig_ejm_42	-15.71	0.03
complex	lig_ejm_31	lig_ejm_46	-40.75	0.04
solvent	lig_ejm_31	lig_ejm_46	-39.86	0.05
complex	lig_ejm_31	lig_ejm_47	-27.8	0.1
solvent	lig_ejm_31	lig_ejm_47	-27.83	0.06
complex	lig_ejm_31	lig_ejm_48	-16.14	0.08
solvent	lig_ejm_31	lig_ejm_48	-16.76	0.03
complex	lig_ejm_31	lig_ejm_50	-57.33	0.04
solvent	lig_ejm_31	lig_ejm_50	-58.33	0.02
complex	lig_ejm_42	lig_ejm_43	-18.9	0.2
solvent	lig_ejm_42	lig_ejm_43	-20.28	0.03
complex	lig_ejm_46	lig_jmc_23	17.42	0.06
solvent	lig_ejm_46	lig_jmc_23	17.12	0.06
complex	lig_ejm_46	lig_jmc_27	15.81	0.09
solvent	lig_ejm_46	lig_jmc_27	15.91	0.05
complex	lig_ejm_46	lig_jmc_28	23.14	0.04
solvent	lig_ejm_46	lig_jmc_28	23.41	0.05
"""


_EXPECTED_RAW = b"""\
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
POOCH_CACHE = pooch.os_cache('openfe')
ZENODO_RBFE_DATA = pooch.create(
        path = POOCH_CACHE,
        base_url="doi:10.5281/zenodo.15042470",
        registry={
            "rbfe_results_serial_repeats.tar.gz": "md5:2355ecc80e03242a4c7fcbf20cb45487",
            "rbfe_results_parallel_repeats.tar.gz": "md5:ff7313e14eb6f2940c6ffd50f2192181"},
        retry_if_failed=3,
    )

@pytest.fixture
def rbfe_result_dir()->pathlib.Path:
    def _rbfe_result_dir(dataset)->str:
        ZENODO_RBFE_DATA.fetch(f'{dataset}.tar.gz', processor=pooch.Untar())
        cache_dir = pathlib.Path(pooch.os_cache('openfe'))/f'{dataset}.tar.gz.untar/{dataset}/'
        return  cache_dir

    return _rbfe_result_dir

@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,reason="Internet seems to be unavailable and test data is not cached locally.")
@pytest.mark.parametrize('dataset', ['rbfe_results_serial_repeats', 'rbfe_results_parallel_repeats'])
@pytest.mark.parametrize('report', ["", "dg", "ddg", "raw"])
@pytest.mark.parametrize('input_mode', ['directory','filepaths'])
def test_gather(rbfe_result_dir, dataset, report, input_mode):

    expected = {
        "": _EXPECTED_DG,
        "dg": _EXPECTED_DG,
        "ddg": _EXPECTED_DDG,
        "raw": _EXPECTED_RAW,
    }[report]
    runner = CliRunner()

    if report:
        args = ["--report", report]
    else:
        args = []

    results = rbfe_result_dir(dataset)
    if input_mode == 'directory':
        results = [str(results)]
    elif input_mode == 'filepaths':
        results = glob.glob(f"{results}/*", recursive=True)
        assert len(results) > 1  # sanity check to make sure we're passing in multiple paths

    cli_result = runner.invoke(gather, results + args + ['-o', '-'])

    assert_click_success(cli_result)

    actual_lines = set(cli_result.stdout_bytes.split(b'\n'))
    assert set(expected.split(b'\n')) == actual_lines

@pytest.mark.skipif(not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,reason="Internet seems to be unavailable and test data is not cached locally.")
class TestGatherFailedEdges:
    @pytest.fixture()
    def results_dir_serial_missing_legs(self, rbfe_result_dir, tmpdir)->str:
        """Example output data, with replicates run in serial and two missing results JSONs."""
        # TODO: update to return a list of paths without doing this symlink mess, when gather supports it.
        rbfe_result_dir = rbfe_result_dir('rbfe_results_serial_repeats')
        tmp_results_dir =  tmpdir
        files_to_skip = ["rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json",
                         "rbfe_lig_ejm_46_solvent_lig_jmc_28_solvent.json"
                            ]
        for item in os.listdir(rbfe_result_dir):
            if item not in files_to_skip:
                os.symlink(rbfe_result_dir/item, tmp_results_dir/item)

        return str(tmp_results_dir)

    def test_missing_leg_error(self, results_dir_serial_missing_legs: str):
        runner = CliRunner()
        result = runner.invoke(gather, [results_dir_serial_missing_legs] + ['-o', '-'])

        assert result.exit_code == 1
        assert isinstance(result.exception, RuntimeError)
        assert "Some edge(s) are missing runs" in str(result.exception)
        assert "('lig_ejm_31', 'lig_ejm_42'): solvent" in str(result.exception)
        assert "('lig_ejm_46', 'lig_jmc_28'): complex" in str(result.exception)
        assert "using the --allow-partial flag" in str(result.exception)


    def test_missing_leg_allow_partial(self, results_dir_serial_missing_legs: str):
        runner = CliRunner()
        # we *dont* want the suggestion to use --allow-partial if the user already used it!
        with pytest.warns(match='[^using the \-\-allow\-partial]'):
            result = runner.invoke(gather, [results_dir_serial_missing_legs] + ['--allow-partial', '-o', '-'])
            assert_click_success(result)
