"""Utilities for building Exorcist task graphs and task databases.

This module translates an :class:`gufe.AlchemicalNetwork` into Exorcist task
structures and can initialize an Exorcist task database from that graph.
"""

from pathlib import Path

import exorcist
import networkx as nx
import pandas as pd
from gufe import AlchemicalNetwork, ProtocolDAG

from ..storage.warehouse import FileSystemWarehouse, WarehouseBaseClass


def _alchemical_network_to_task_graph(
    alchemical_network: AlchemicalNetwork,
    warehouse: WarehouseBaseClass,
) -> nx.DiGraph:
    """Build a global task DAG from `alchemical_network` and store its relevant data
    in `warehouse` the following warehouse stores:
        - 'setup': The AlchemicalNetwork, deduplicated on disk
        - 'tasks': The ProtocolUnits to be executed as tasks
        - 'protocol_dags': The ProtocolDAGs that the ProtocolUnits belong to.
                           Used to gather results after execution.

    Parameters
    ----------
    alchemical_network : AlchemicalNetwork
        Network containing alchemical Transformations to be executed.
    warehouse : WarehouseBaseClass
        Warehouse used to store data used by the execution and simulation engines.

    Returns
    -------
    nx.DiGraph
        A directed acyclic graph where each node is a task ID with the ProtocolUnit key as a name
        and edges encode protocol-unit dependencies.

    Raises
    ------
    ValueError
        If the assembled task graph is not acyclic.
        If the input `alchemical_network` is not a valid openfe.AlchemicalNetwork
    """

    if not isinstance(alchemical_network, AlchemicalNetwork):
        raise ValueError(
            f"alchemical_network must be an AlchemicalNetwork, not {type(alchemical_network)}."
        )

    warehouse.store_setup_tokenizable(alchemical_network)

    global_task_dag = nx.DiGraph()
    for transformation in alchemical_network.edges:
        dag: ProtocolDAG = transformation.create()  # TODO: skip edges that already have units?
        for unit in dag.protocol_units:
            global_task_dag.add_node(str(unit.key))
            warehouse.store_task(unit)
        for dependent_unit, dependency_unit in dag.graph.edges:
            upstream_id = str(dependency_unit.key)
            downstream_id = str(dependent_unit.key)
            global_task_dag.add_edge(upstream_id, downstream_id)

        # at this point, stored as a shallow dict since all its units are already stored
        warehouse.store_protocol_dag(dag)

    if not nx.is_directed_acyclic_graph(global_task_dag):
        raise ValueError("AlchemicalNetwork produced a task graph that is not a DAG.")

    return global_task_dag


# TODO: add n_repeats
def build_task_db_from_alchemical_network(
    alchemical_network: AlchemicalNetwork,
    warehouse_dir: Path,  # TODO: make optional?
    db_path: Path,  # TODO: make optional?
    max_tries: int = 1,
) -> tuple[exorcist.TaskStatusDB, FileSystemWarehouse]:
    """Create a task database and FileSystemWarehouse from an alchemical network.

    Parameters
    ----------
    alchemical_network : AlchemicalNetwork
        ``AlchemicalNetwork`` containing transformations to convert into task records.
    warehouse_dir : pathlib.Path
        Root directory at which to create a FileSystemWarehouse (e.g. ``campaign_name/``).
    db_path : pathlib.Path
        Location to store the SQLite-backed Exorcist database (e.g. ``campaign_name.db``).
    max_tries : int, default=1
        Maximum number of times a task will attempt to be submitted before it
        is labelled ``TOO_MANY_RETRIES``.

    Returns
    -------
    exorcist.TaskStatusDB
        Initialized task database populated with graph nodes and dependency
        edges derived from ``alchemical_network``.
    """
    # require starting clean each time for now - guardrails around modifying existing state can come later
    if Path(db_path).exists():
        raise FileExistsError(f"Error: {db_path} cannot already exist.")
    warehouse = FileSystemWarehouse(warehouse_dir, exist_ok=False)
    global_task_dag: nx.DiGraph = _alchemical_network_to_task_graph(alchemical_network, warehouse)
    db = exorcist.TaskStatusDB.from_filename(db_path)
    db.add_task_network(global_task_dag, max_tries)
    return db, warehouse


def get_task_df(task_db: exorcist.TaskStatusDB) -> pd.DataFrame:
    """Create a pandas Dataframe from task_db.

    Parameters
    ----------
    task_db : exorcist.TaskStatusDB
        A task database.

    Returns
    -------
    pd.DataFrame
        A dataframe of the tasks and their statuses
    """
    status_name_encoding = {e.value: e.name for e in exorcist.TaskStatus}
    task_table = pd.read_sql_table("tasks", task_db.engine)
    task_table.replace({"status": status_name_encoding}, inplace=True)
    return task_table


def get_dependency_df(task_db: exorcist.TaskStatusDB) -> pd.DataFrame:
    """Create a pandas Dataframe from task_db.

    Parameters
    ----------
    task_db : exorcist.TaskStatusDB
        A task database.

    Returns
    -------
    pd.DataFrame
        A dataframe of the tasks and their dependencies.

    """
    return pd.read_sql_table("dependencies", task_db.engine)
