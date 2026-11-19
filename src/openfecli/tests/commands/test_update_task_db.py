import exorcist
import networkx as nx
import pytest
import sqlalchemy as sqla
from click.testing import CliRunner

from openfecli.commands.update_task_db import update_task_db

from ..utils import assert_click_success


@pytest.fixture
def simple_task_graph():
    task_graph = nx.DiGraph()
    node_ids = ["HybridTopologySetupUnit-123", "HybridTopologySetupUnit-456"]

    for id in node_ids:
        task_graph.add_node(id)

    return task_graph, node_ids


def test_update_task_db(simple_task_graph):
    runner = CliRunner()
    with runner.isolated_filesystem():
        task_graph, _ = simple_task_graph
        db_path = "test.db"
        db = exorcist.TaskStatusDB.from_filename(db_path)
        db.add_task_network(task_graph, max_tries=2)
        input_max_tries = 7
        result = runner.invoke(
            update_task_db, ["--task-db", db_path, "--max-tries", str(input_max_tries)]
        )
        assert_click_success(result)

        # just make sure the table is actually modified. all edge cases etc. are handled in the Python API tests
        with db.engine.connect() as conn:
            new_max_tries = set(conn.execute(sqla.select(db.tasks_table.c.max_tries)))

        assert new_max_tries == {(input_max_tries,)}
