from click.testing import CliRunner
import glob
from importlib import resources
import tarfile
import pytest

from openfecli.commands.gather import gather


@pytest.fixture
def results_dir(tmpdir):
    with tmpdir.as_cwd():
        with resources.path('openfecli.tests.data', 'results.tar.gz') as f:
            t = tarfile.open(f, mode='r')
            t.extractall('.')

        yield


@pytest.fixture
def ref_gather():
    return b"""
measurement\ttype\tligand_i\tligand_j\testimate (kcal/mol)\tuncertainty (kcal/mol)
DDGhyd(lig_15, lig_12)\tRHFE\tlig_12\tlig_15\t-1.1\t0.055
DDGhyd(lig_16, lig_15)\tRHFE\tlig_15\tlig_16\t-5.7\t0.043
DDGhyd(lig_15, lig_11)\tRHFE\tlig_11\tlig_15\t5.4\t 0.045
DDGhyd(lig_3, lig_14)\tRHFE\tlig_14\tlig_3\t1.3\t 0.12
DDGhyd(lig_6, lig_1)\tRHFE\tlig_1\tlig_6\t-3.5\t0.038
DDGhyd(lig_8, lig_6)\tRHFE\tlig_6\tlig_8\t4.1\t 0.074
DDGhyd(lig_3, lig_2)\tRHFE\tlig_2\tlig_3\t0.23\t0.044
DDGhyd(lig_5, lig_10)\tRHFE\tlig_10\tlig_5\t-8.8\t0.099
DDGhyd(lig_15, lig_10)\tRHFE\tlig_10\tlig_15\t1.4\t 0.047
DDGhyd(lig_6, lig_14)\tRHFE\tlig_14\tlig_6\t-2.1\t0.034
DDGhyd(lig_15, lig_14)\tRHFE\tlig_14\tlig_15\t3.3\t 0.056
DDGhyd(lig_9, lig_6)\tRHFE\tlig_6\tlig_9\t-0.079\t0.021
DDGhyd(lig_7, lig_3)\tRHFE\tlig_3\tlig_7\t73.0\t2.0
DDGhyd(lig_7, lig_4)\tRHFE\tlig_4\tlig_7\t2.7\t 0.16
DDGhyd(lig_14, lig_13)\tRHFE\tlig_13\tlig_14\t0.49\t0.038
DGsolvent(lig_12, lig_15)\tsolvent lig_12\tlig_15\t-5.1\t0.055
DGvacuum(lig_12, lig_15)\tvacuum\tlig_12\tlig_15\t-4.0\t0.0014
DGsolvent(lig_15, lig_16)\tsolvent lig_15\tlig_16\t-17.0\t0.043
DGvacuum(lig_15, lig_16)\tvacuum\tlig_15\tlig_16\t-11.0\t0.0064
DGsolvent(lig_11, lig_15)\tsolvent lig_11\tlig_15\t4.1\t 0.041
DGvacuum(lig_11, lig_15)\tvacuum\tlig_11\tlig_15\t-1.3\t0.019
DGvacuum(lig_14, lig_3)\tvacuum\tlig_14\tlig_3\t-29.0\t0.051
DGsolvent(lig_14, lig_3)\tsolvent lig_14\tlig_3\t-28.0\t0.11
DGvacuum(lig_1, lig_6)\tvacuum\tlig_1\tlig_6\t20.0\t0.022
DGsolvent(lig_1, lig_6)\tsolvent lig_1\tlig_6\t17.0\t0.032
DGsolvent(lig_6, lig_8)\tsolvent lig_6\tlig_8\t-6.1\t0.069
DGvacuum(lig_6, lig_8)\tvacuum\tlig_6\tlig_8\t-10.0\t0.027
DGsolvent(lig_2, lig_3)\tsolvent lig_2\tlig_3\t15.0\t0.03
DGvacuum(lig_2, lig_3)\tvacuum\tlig_2\tlig_3\t15.0\t0.032
DGvacuum(lig_10, lig_5)\tvacuum\tlig_10\tlig_5\t19.0\t0.046
DGsolvent(lig_10, lig_5)\tsolvent lig_10\tlig_5\t11.0\t0.087
DGvacuum(lig_10, lig_15)\tvacuum\tlig_10\tlig_15\t2.3\t 0.01
DGsolvent(lig_10, lig_15)\tsolvent lig_10\tlig_15\t3.7\t 0.046
DGvacuum(lig_14, lig_6)\tvacuum\tlig_14\tlig_6\t16.0\t0.011
DGsolvent(lig_14, lig_6)\tsolvent lig_14\tlig_6\t14.0\t0.032
DGsolvent(lig_14, lig_15)\tsolvent lig_14\tlig_15\t10.0\t0.056
DGvacuum(lig_14, lig_15)\tvacuum\tlig_14\tlig_15\t6.9\t 0.0028
DGvacuum(lig_6, lig_9)\tvacuum\tlig_6\tlig_9\t-5.0\t0.00056
DGsolvent(lig_6, lig_9)\tsolvent lig_6\tlig_9\t-5.1\t0.021
DGvacuum(lig_3, lig_7)\tvacuum\tlig_3\tlig_7\t-28.0\t0.91
DGsolvent(lig_3, lig_7)\tsolvent lig_3\tlig_7\t45.0\t1.8
DGsolvent(lig_4, lig_7)\tsolvent lig_4\tlig_7\t-3.3\t0.15
DGvacuum(lig_4, lig_7)\tvacuum\tlig_4\tlig_7\t-6.1\t0.048
DGsolvent(lig_13, lig_14)\tsolvent lig_13\tlig_14\t15.0\t0.037
DGvacuum(lig_13, lig_14)\tvacuum\tlig_13\tlig_14\t15.0\t0.0057
"""


def test_gather(results_dir, ref_gather):
    runner = CliRunner()

    result = runner.invoke(gather, ['results', '-o', '-'])

    assert result.exit_code == 0

    actual_lines = set(result.stdout_bytes.split(b'\n'))

    assert set(ref_gather.split(b'\n')) == actual_lines

