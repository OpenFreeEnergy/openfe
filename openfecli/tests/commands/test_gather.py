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
    return b"""\
measurement\testimate (kcal/mol)\tuncertainty
DDGhyd(lig_15, lig_12)\t-1.1\t+-0.055
DDGhyd(lig_8, lig_6)\t4.1\t+-0.074
DDGhyd(lig_3, lig_2)\t0.23\t+-0.044
DDGhyd(lig_15, lig_10)\t1.4\t+-0.047
DDGhyd(lig_5, lig_10)\t-8.8\t+-0.099
DDGhyd(lig_3, lig_14)\t1.3\t+-0.12
DDGhyd(lig_15, lig_14)\t3.3\t+-0.056
DDGhyd(lig_15, lig_11)\t5.4\t+-0.045
DDGhyd(lig_9, lig_6)\t-0.079\t+-0.021
DDGhyd(lig_6, lig_14)\t-2.1\t+-0.034
DDGhyd(lig_14, lig_13)\t0.49\t+-0.038
DDGhyd(lig_7, lig_3)\t73.0\t+-2.0
DDGhyd(lig_6, lig_1)\t-3.5\t+-0.038
DDGhyd(lig_16, lig_15)\t-5.7\t+-0.043
DDGhyd(lig_7, lig_4)\t2.7\t+-0.16
DGsolvent(lig_12, lig_15)\t-5.1\t+-0.055
DGvacuum(lig_12, lig_15)\t-4.0\t+-0.0014
DGvacuum(lig_6, lig_8)\t-10.0\t+-0.027
DGsolvent(lig_6, lig_8)\t-6.1\t+-0.069
DGsolvent(lig_2, lig_3)\t15.0\t+-0.03
DGvacuum(lig_2, lig_3)\t15.0\t+-0.032
DGvacuum(lig_10, lig_15)\t2.3\t+-0.01
DGsolvent(lig_10, lig_15)\t3.7\t+-0.046
DGvacuum(lig_10, lig_5)\t19.0\t+-0.046
DGsolvent(lig_10, lig_5)\t11.0\t+-0.087
DGsolvent(lig_14, lig_3)\t-28.0\t+-0.11
DGvacuum(lig_14, lig_3)\t-29.0\t+-0.051
DGvacuum(lig_14, lig_15)\t6.9\t+-0.0028
DGsolvent(lig_14, lig_15)\t10.0\t+-0.056
DGsolvent(lig_11, lig_15)\t4.1\t+-0.041
DGvacuum(lig_11, lig_15)\t-1.3\t+-0.019
DGvacuum(lig_6, lig_9)\t-5.0\t+-0.00056
DGsolvent(lig_6, lig_9)\t-5.1\t+-0.021
DGvacuum(lig_14, lig_6)\t16.0\t+-0.011
DGsolvent(lig_14, lig_6)\t14.0\t+-0.032
DGvacuum(lig_13, lig_14)\t15.0\t+-0.0057
DGsolvent(lig_13, lig_14)\t15.0\t+-0.037
DGsolvent(lig_3, lig_7)\t45.0\t+-1.8
DGvacuum(lig_3, lig_7)\t-28.0\t+-0.91
DGsolvent(lig_1, lig_6)\t17.0\t+-0.032
DGvacuum(lig_1, lig_6)\t20.0\t+-0.022
DGvacuum(lig_15, lig_16)\t-11.0\t+-0.0064
DGsolvent(lig_15, lig_16)\t-17.0\t+-0.043
DGvacuum(lig_4, lig_7)\t-6.1\t+-0.048
DGsolvent(lig_4, lig_7)\t-3.3\t+-0.15
"""


def test_gather(results_dir, ref_gather):
    runner = CliRunner()

    result = runner.invoke(gather, ['results', '-o', '-'])

    assert result.exit_code == 0

    actual_lines = set(result.stdout_bytes.split(b'\n'))

    assert set(ref_gather.split(b'\n')) == actual_lines

