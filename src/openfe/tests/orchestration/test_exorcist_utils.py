from pathlib import Path
from unittest import mock

import exorcist
import networkx as nx
import pytest
import sqlalchemy as sqla
from gufe.tokenization import GufeKey

from openfe.orchestration.exorcist_utils import (
    alchemical_network_to_task_graph,
    build_task_db_from_alchemical_network,
)
from openfe.storage.warehouse import FileSystemWarehouse


class _RecordingWarehouse:
    def __init__(self):
        self.stored_tasks = []

    def store_task(self, task):
        self.stored_tasks.append(task)


def _network_units(benzene_variants_star_map):
    units = []
    for transformation in benzene_variants_star_map.edges:
        units.extend(transformation.create().protocol_units)
    return units


@pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
def test_alchemical_network_to_task_graph_stores_all_units(request, fixture):
    warehouse = _RecordingWarehouse()
    network = request.getfixturevalue(fixture)
    expected_units = _network_units(network)

    alchemical_network_to_task_graph(network, warehouse)

    stored_unit_names = [str(unit.name) for unit in warehouse.stored_tasks]
    expected_unit_names = [str(unit.name) for unit in expected_units]

    assert len(stored_unit_names) == len(expected_unit_names)
    assert sorted(stored_unit_names) == sorted(expected_unit_names)


@pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
def test_alchemical_network_to_task_graph_uses_canonical_task_ids(request, fixture):
    warehouse = _RecordingWarehouse()
    network = request.getfixturevalue(fixture)

    graph = alchemical_network_to_task_graph(network, warehouse)

    transformation_keys = {str(transformation.key) for transformation in network.edges}
    expected_protocol_unit_keys = sorted(str(unit.key) for unit in warehouse.stored_tasks)
    observed_protocol_unit_keys = []

    for node in graph.nodes:
        transformation_key, protocol_unit_key = node.split(":", maxsplit=1)
        assert transformation_key in transformation_keys
        observed_protocol_unit_keys.append(protocol_unit_key)

    assert sorted(observed_protocol_unit_keys) == expected_protocol_unit_keys


@pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
def test_alchemical_network_to_task_graph_edges_reference_existing_nodes(request, fixture):
    warehouse = _RecordingWarehouse()
    network = request.getfixturevalue(fixture)

    graph = alchemical_network_to_task_graph(network, warehouse)

    assert len(graph.edges) > 0
    for u, v in graph.edges:
        assert u in graph.nodes
        assert v in graph.nodes


def test_alchemical_network_to_task_graph_raises_for_cycle():
    class _Unit:
        def __init__(self, name: str, key: str):
            self.name = name
            self.key = key

    class _Transformation:
        name = "cyclic"
        key = "Transformation-cycle"

        def create(self):
            unit_a = _Unit("unit-a", "ProtocolUnit-a")
            unit_b = _Unit("unit-b", "ProtocolUnit-b")
            dag = mock.Mock()
            dag.protocol_units = [unit_a, unit_b]
            dag.graph = nx.DiGraph()
            dag.graph.add_nodes_from([unit_a, unit_b])
            dag.graph.add_edges_from([(unit_a, unit_b), (unit_b, unit_a)])
            return dag

    network = mock.Mock()
    network.edges = [_Transformation()]
    warehouse = mock.Mock()

    with pytest.raises(ValueError, match="not a DAG"):
        alchemical_network_to_task_graph(network, warehouse)


@pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
def test_build_task_db_checkout_order_is_dependency_safe(tmp_path, request, fixture):
    network = request.getfixturevalue(fixture)
    warehouse = FileSystemWarehouse(str(tmp_path / "warehouse"))
    # Build the real sqlite task DB from a real alchemical network fixture.
    db = build_task_db_from_alchemical_network(
        network,
        warehouse,
        db_path=tmp_path / "tasks.db",
    )

    # Read task IDs and dependency edges from the persisted DB state.
    initial_task_rows = list(db.get_all_tasks())
    graph_taskids = {row.taskid for row in initial_task_rows}
    with db.engine.connect() as conn:
        dep_rows = conn.execute(sqla.select(db.dependencies_table)).all()
    graph_edges = {(row._mapping["from"], row._mapping["to"]) for row in dep_rows}

    checkout_order = []
    # Hard upper bound prevents infinite checkout loops.
    max_checkouts = len(graph_taskids)
    print(f"Max Checkout={max_checkouts}")
    for _ in range(max_checkouts):
        taskid = db.check_out_task()
        if taskid is None:
            break

        checkout_order.append(taskid)
        _, protocol_unit_key = taskid.split(":", maxsplit=1)
        loaded_unit = warehouse.load_task(GufeKey(protocol_unit_key))
        assert str(loaded_unit.key) == protocol_unit_key
        db.mark_task_completed(taskid, success=True)

    # Coverage/completion: every task is checked out exactly once.
    observed_taskids = set(checkout_order)
    assert observed_taskids == graph_taskids
    assert len(checkout_order) == len(graph_taskids)

    # Dependency safety: upstream tasks must appear before downstream tasks.
    checkout_index = {taskid: idx for idx, taskid in enumerate(checkout_order)}
    for upstream, downstream in graph_edges:
        assert checkout_index[upstream] < checkout_index[downstream]

    # Final DB state: all tasks are completed.
    task_rows = list(db.get_all_tasks())
    assert len(task_rows) == len(graph_taskids)
    assert {row.taskid for row in task_rows} == graph_taskids
    assert {row.status for row in task_rows} == {exorcist.TaskStatus.COMPLETED.value}


@pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
def test_build_task_db_default_path(request, fixture):
    network = request.getfixturevalue(fixture)
    warehouse = mock.Mock()
    fake_graph = nx.DiGraph()
    fake_db = mock.Mock()

    with (
        mock.patch(
            "openfe.orchestration.exorcist_utils.alchemical_network_to_task_graph",
            return_value=fake_graph,
        ) as task_graph_mock,
        mock.patch(
            "openfe.orchestration.exorcist_utils.exorcist.TaskStatusDB.from_filename",
            return_value=fake_db,
        ) as db_ctor,
    ):
        result = build_task_db_from_alchemical_network(network, warehouse)

    task_graph_mock.assert_called_once_with(network, warehouse)
    db_ctor.assert_called_once_with(Path("tasks.db"))
    fake_db.add_task_network.assert_called_once_with(fake_graph, 1)
    assert result is fake_db


@pytest.mark.parametrize("fixture", ["benzene_variants_star_map"])
def test_build_task_db_forwards_graph_and_max_tries(request, tmp_path, fixture):
    network = request.getfixturevalue(fixture)
    warehouse = mock.Mock()
    fake_graph = nx.DiGraph()
    fake_db = mock.Mock()
    db_path = tmp_path / "custom_tasks.db"

    with (
        mock.patch(
            "openfe.orchestration.exorcist_utils.alchemical_network_to_task_graph",
            return_value=fake_graph,
        ) as task_graph_mock,
        mock.patch(
            "openfe.orchestration.exorcist_utils.exorcist.TaskStatusDB.from_filename",
            return_value=fake_db,
        ) as db_ctor,
    ):
        result = build_task_db_from_alchemical_network(
            network,
            warehouse,
            db_path=db_path,
            max_tries=7,
        )

    task_graph_mock.assert_called_once_with(network, warehouse)
    db_ctor.assert_called_once_with(db_path)
    fake_db.add_task_network.assert_called_once_with(fake_graph, 7)
    assert result is fake_db
