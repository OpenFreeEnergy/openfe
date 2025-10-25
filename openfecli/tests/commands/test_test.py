import pytest
from unittest import mock
from click.testing import CliRunner
import os

from openfecli.commands.test import test


def mock_func(args):
    print(os.environ.get("OFE_SLOW_TESTS"))


@pytest.mark.parametrize("slow", [True, False])
def test_test(slow):
    runner = CliRunner()
    args = ["--long"] if slow else []
    patchloc = "openfecli.commands.test.pytest.main"
    ofe_slow_tests = os.environ.get("OFE_SLOW_TESTS")
    with mock.patch(patchloc, mock_func):
        with runner.isolated_filesystem():
            result = runner.invoke(test, args)
            assert result.exit_code == 0
            l1, l2, l3, _ = result.output.split("\n")
            assert l1 == "Testing can import...."
            assert l2 == "Running the main package tests"
            assert l3 == str(slow)

    assert ofe_slow_tests == os.environ.get("OFE_SLOW_TESTS")
