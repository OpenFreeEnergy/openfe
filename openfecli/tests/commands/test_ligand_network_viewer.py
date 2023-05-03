import pytest
from click.testing import CliRunner
import importlib.resources
import matplotlib

from openfecli.commands.ligand_network_viewer import ligand_network_viewer

def test_ligand_network_viewer():
    # smoke test
    resource = importlib.resources.files('openfe.tests.data.serialization')
    ref = resource / "network_template.graphml"
    runner = CliRunner()
    with runner.isolated_filesystem():
        # interestingly, this doesn't seem to require the patch to avoid
        # launching a matplotlib window
        runner.invoke(ligand_network_viewer, ref)
