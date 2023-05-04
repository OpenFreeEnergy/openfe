import pytest
from unittest import mock
from click.testing import CliRunner
import importlib.resources
import matplotlib

from openfecli.commands.ligand_network_viewer import ligand_network_viewer

@pytest.mark.filterwarnings("ignore:.*non-GUI backend")
def test_ligand_network_viewer():
    # smoke test
    resource = importlib.resources.files('openfe.tests.data.serialization')
    ref = resource / "network_template.graphml"
    runner = CliRunner()

    backend = matplotlib.get_backend()
    matplotlib.use("ps")
    loc = "openfe.utils.atommapping_network_plotting.matplotlib.use"
    with runner.isolated_filesystem():
        with mock.patch(loc, mock.Mock()):
            result = runner.invoke(ligand_network_viewer, [str(ref)])
            assert result.exit_code == 0

    matplotlib.use(backend)
