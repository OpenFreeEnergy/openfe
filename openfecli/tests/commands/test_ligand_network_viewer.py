import importlib.resources
from unittest import mock

import matplotlib
import pytest
from click.testing import CliRunner

from openfecli.commands.view_ligand_network import view_ligand_network


@pytest.mark.filterwarnings("ignore:.*non-GUI backend")
def test_view_ligand_network():
    # smoke test
    resource = importlib.resources.files("openfe.tests.data.serialization")
    ref = resource / "network_template.graphml"
    runner = CliRunner()

    backend = matplotlib.get_backend()
    matplotlib.use("ps")
    loc = "openfe.utils.atommapping_network_plotting.matplotlib.use"
    with runner.isolated_filesystem():
        with mock.patch(loc, mock.Mock()):
            result = runner.invoke(view_ligand_network, [str(ref)])
            assert result.exit_code == 0

    matplotlib.use(backend)
