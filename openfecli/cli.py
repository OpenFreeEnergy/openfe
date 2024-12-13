# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib
import logging
import logging.config

import click
from plugcli.cli import CLI, CONTEXT_SETTINGS
from plugcli.plugin_management import FilePluginLoader

import openfecli

from openfecli.plugins import OFECommandPlugin


class OpenFECLI(CLI):
    # COMMAND_SECTIONS = ["Setup", "Simulation", "Orchestration", "Analysis"]
    COMMAND_SECTIONS = ["Network Planning", "Quickrun Executor",
                        "Miscellaneous"]

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
@click.option('--log', type=click.Path(exists=True, readable=True),
              help="logging configuration file")
def main(log):
    # Subcommand runs after this is processed.
    # set logging if provided
    if log:
        logging.config.fileConfig(log, disable_existing_loggers=False)


if __name__ == "__main__":  # -no-cov- (useful in debugging)
    main()
