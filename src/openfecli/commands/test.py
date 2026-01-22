import os
import pathlib
import sys

import click
import pooch
import pytest

from openfecli import OFECommandPlugin
from openfecli.utils import write


def retrieve_test_data():
    pass


@click.command("test", short_help="Run the OpenFE test suite")
@click.option('--long', is_flag=True, default=False, help="Run additional tests (takes much longer)")  # fmt: skip
@click.option(
    "--download-only",
    is_flag=True,
    default=False,
    help="Download data to the cache (this is helpful if internet is flaky).",
)
def test(long, download_only):
    """
    Run the OpenFE test suite. This first checks that OpenFE is correctly
    imported, and then runs the main test suite, which should take several
    minutes. If given the ``--long`` flag, this will include several tests
    that take significantly longer, but ensure that we're able to fully run
    in your environment.

    A successful run will include tests that pass, skip, or "xfail". In many
    terminals, these show as green or yellow. Warnings are not a concern.
    However, You should not see anything that fails or errors (red).
    """
    if download_only:
        retrieve_test_data()
        sys.exit()
    try:
        old_env = dict(os.environ)
        os.environ["OFE_SLOW_TESTS"] = str(long)

        write("Testing can import....")
        import openfe

        write("Running the main package tests")
        return_value = pytest.main(["-v", "--pyargs", "openfe", "--pyargs", "openfecli"])
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        sys.exit(return_value)


PLUGIN = OFECommandPlugin(test, "Miscellaneous", requires_ofe=(0, 7, 5))
