# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import urllib
import shutil
from plugcli.cli import CLI, CONTEXT_SETTINGS
from openfecli.fetching import FetchablePlugin
from openfecli import OFECommandPlugin

# MOVE SINGLEMODULEPLUGINLOADER UPSTREAM TO PLUGCLI
import importlib
from plugcli.plugin_management import CLIPluginLoader
class SingleModulePluginLoader(CLIPluginLoader):
    """Load plugins from a specific module
    """
    def __init__(self, module_name, plugin_class):
        super().__init__(plugin_type="single_module",
                         search_path=module_name,
                         plugin_class=plugin_class)

    def _find_candidates(self):
        return [importlib.import_module(self.search_path)]

    @staticmethod
    def _make_nsdict(candidate):
        return vars(candidate)


class FetchCLI(CLI):
    """Custom command class for the Fetch subcommand.

    This provides the command sections used in help and defines where
    plugins should be kept.
    """
    COMMAND_SECTIONS = ["Built-in", "Requires Internet"]

    def get_loaders(self):
        return [
            SingleModulePluginLoader('openfecli.fetchables',
                                     FetchablePlugin)
        ]

    def get_installed_plugins(self):
        loader = self.get_loaders()[0]
        return list(loader())

@click.command(
    cls=FetchCLI,
    short_help="Fetch tutorial or other resource."
)
def fetch():
    """
    Fetch the given resource. Some resources require internet; others are
    built-in.
    """

PLUGIN = OFECommandPlugin(
    command=fetch,
    section="Miscellaneous",
    requires_ofe=(0, 7),
)

if __name__ == "__main__":
    # it's useful to keep a main here for debugging where problems happen in
    # the command tree
    fetch()
