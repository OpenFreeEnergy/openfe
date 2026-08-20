import exorcist
import networkx as nx
import pytest
from click.testing import CliRunner

from openfecli.commands.status import status

from ..utils import assert_click_success


@pytest.fixture
def single_task_graph():
    task_graph = nx.DiGraph()
    node_ids = ["HybridTopologySetupUnit-123", "HybridTopologySetupUnit-456"]

    for id in node_ids:
        task_graph.add_node(id)

    return task_graph, node_ids


def test_status(single_task_graph):
    runner = CliRunner()
    with runner.isolated_filesystem():
        task_graph, node_ids = single_task_graph
        db_path = "test.db"
        db = exorcist.TaskStatusDB.from_filename(db_path)
        db.add_task_network(task_graph, max_tries=6)
        result = runner.invoke(status, [db_path])
        assert_click_success(result)
        assert all(id in result.stdout for id in node_ids)
