# THIS IS A TEMPORARY FILE TO ILLUSTRATE HOW ONE WRITES A SUBCOMMAND PLUGIN.
# DELETE ONCE WE HAVE A FEW REAL EXAMPLES!

import click
from openfecli import OFECommandPlugin

@click.command(
    "echo",
    short_help="This shows up in ``openfe --help``"
)
def echo():
    """
    This is the longer help statement that shows up when you get help for an
    individual command, e.g., with ``openfe echo --help``.
    """
    # the code here serves to convert user input to objects that would be
    # run by library code. In general, this should be done with a ``get``
    # method attached to the input decorators
    echo_main()

def echo_main():
    # the code here does the actual workflow in the library. This will tend
    # to be a very small code
    print("foo")


PLUGIN = OFECommandPlugin(
    command=echo,
    section="Simulation",
    requires_ofe=(0, 0, 1)
)

if __name__ == "__main__":
    echo()
