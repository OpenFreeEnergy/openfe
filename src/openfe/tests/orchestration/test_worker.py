from pathlib import Path
from unittest import mock

import exorcist
import gufe
import networkx as nx
import pytest
from gufe.protocols.protocolunit import ProtocolUnit

from openfe.orchestration import Worker
from openfe.orchestration.exorcist_utils import build_task_db_from_alchemical_network
from openfe.storage.warehouse import FileSystemWarehouse


def _result_store_files(warehouse: FileSystemWarehouse) -> set[str]:
    result_root = Path(warehouse.result_store.root_dir)
    return {str(path.relative_to(result_root)) for path in result_root.rglob("*") if path.is_file()}


def _contains_protocol_unit(value) -> bool:
    if isinstance(value, ProtocolUnit):
        return True
    if isinstance(value, dict):
        return any(_contains_protocol_unit(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_protocol_unit(item) for item in value)
    return False


def _get_dependency_free_unit(absolute_transformation):
    for unit in absolute_transformation.create().protocol_units:
        if not _contains_protocol_unit(unit.inputs):
            return unit
    raise ValueError("No dependency-free protocol unit found for execution test setup.")


@pytest.fixture
def worker_with_real_db(tmp_path, absolute_transformation):
    warehouse_root = tmp_path / "warehouse"
    db_path = warehouse_root / "tasks.db"
    warehouse = FileSystemWarehouse(str(warehouse_root))
    network = gufe.AlchemicalNetwork([absolute_transformation])
    db = build_task_db_from_alchemical_network(network, warehouse, db_path=db_path)
    worker = Worker(warehouse=warehouse, task_db_path=db_path)
    return worker, warehouse, db


@pytest.fixture
def worker_with_executable_task_db(tmp_path, absolute_transformation):
    warehouse_root = tmp_path / "warehouse"
    db_path = warehouse_root / "tasks.db"
    warehouse = FileSystemWarehouse(str(warehouse_root))
    unit = _get_dependency_free_unit(absolute_transformation)
    warehouse.store_task(unit)

    taskid = f"{absolute_transformation.key}:{unit.key}"
    task_graph = nx.DiGraph()
    task_graph.add_node(taskid)

    db = exorcist.TaskStatusDB.from_filename(db_path)
    db.add_task_network(task_graph, 1)

    worker = Worker(warehouse=warehouse, task_db_path=db_path)
    return worker, warehouse, db, unit


def test_get_task_uses_default_db_path_without_patching(
    tmp_path, monkeypatch, absolute_transformation
):
    monkeypatch.chdir(tmp_path)
    warehouse = FileSystemWarehouse("warehouse")
    db_path = Path("warehouse/tasks.db")
    network = gufe.AlchemicalNetwork([absolute_transformation])
    db = build_task_db_from_alchemical_network(network, warehouse, db_path=db_path)

    worker = Worker(warehouse=warehouse)
    loaded = worker._get_task()

    expected_keys = {task_row.taskid.split(":", maxsplit=1)[1] for task_row in db.get_all_tasks()}
    assert worker.task_db_path == Path("./warehouse/tasks.db")
    assert str(loaded.key) in expected_keys


def test_get_task_returns_task_with_canonical_protocol_unit_suffix(worker_with_real_db):
    worker, warehouse, db = worker_with_real_db

    task_ids = [row.taskid for row in db.get_all_tasks()]
    expected_protocol_unit_keys = {task_id.split(":", maxsplit=1)[1] for task_id in task_ids}

    loaded = worker._get_task()
    reloaded = warehouse.load_task(loaded.key)

    assert str(loaded.key) in expected_protocol_unit_keys
    assert loaded == reloaded


def test_execute_unit_stores_real_result(worker_with_executable_task_db, tmp_path):
    worker, warehouse, _, _ = worker_with_executable_task_db
    before = _result_store_files(warehouse)

    worker.execute_unit(scratch=tmp_path / "scratch")

    after = _result_store_files(warehouse)
    assert len(after) > len(before)


def test_execute_unit_propagates_execute_error_without_store(
    worker_with_executable_task_db, tmp_path
):
    worker, warehouse, _, unit = worker_with_executable_task_db
    before = _result_store_files(warehouse)

    with mock.patch.object(
        type(unit),
        "execute",
        autospec=True,
        side_effect=RuntimeError("unit execution failed"),
    ):
        with pytest.raises(RuntimeError, match="unit execution failed"):
            worker.execute_unit(scratch=tmp_path / "scratch")

    after = _result_store_files(warehouse)
    assert after == before
