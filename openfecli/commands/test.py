import click
from openfecli import OFECommandPlugin

import pytest
import os

from openfecli.utils import write

@click.command(
    "test",
    short_help="Run the OpenFE test suite"
)
@click.option('--long', is_flag=True, default=False,
              help="Run additional tests (takes much longer)")
def test(long):
    """
    Run the OpenFE test suite. This first checks that OpenFE is correctly
    imported, and then runs the main test suite, which should take several
    minutes. If given the ``--long`` flag, this will include several tests
    that take significantly longer, but ensure that we're able to fully run
    in your environment.
    """
    default = ["-v"]
    pyargs = ["--pyargs", "openfe", "--pyargs", "openfecli"]
    old_env = dict(os.environ)
    os.environ["OFE_SLOW_TESTS"] = long

    write("Testing can import....")
    import openfe
    write("Running the main package tests")
    pytest.main(["-v", "--pyargs", "openfe", "--pyargs", "openfecli"])

    os.environ.clear()
    os.environ.update(old_env)

PLUGIN = OFECommandPlugin(
    test,
    "hidden",
    requires_ofe=(0, 7,5)
)
