# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib

import click
from plugcli.cli import CLI, CONTEXT_SETTINGS
from plugcli.plugin_management import FilePluginLoader

import openfecli

from openfecli.plugins import OFECommandPlugin


class OpenFECLI(CLI):
    COMMAND_SECTIONS = ["Setup", "Simulation", "Orchestration", "Analysis"]

    def get_loaders(self):
        commands = str(pathlib.Path(__file__).parent.resolve() / "commands")
        loader = FilePluginLoader(commands, OFECommandPlugin)
        return [loader]

    def get_installed_plugins(self):
        loader = self.get_loaders()[0]
        return list(loader())


_MAIN_HELP = """
This is the command line tool to provide easy access to functionality from
the OpenFE Python library.
"""


@click.command(cls=OpenFECLI, name="openfe", help=_MAIN_HELP,
               context_settings=CONTEXT_SETTINGS)
@click.version_option(version=openfecli.__version__)
def main():
    # currently empty: we can add options at the openfe level (as opposed to
    # subcommands) by adding click options here. Subcommand runs after this
    # is the processed.
    pass


if __name__ == "__main__":  # -no-cov- (useful in debugging)
    main()
