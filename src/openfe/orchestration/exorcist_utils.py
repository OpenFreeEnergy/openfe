"""Utilities for building Exorcist task graphs and task databases.

This module translates an :class:`gufe.AlchemicalNetwork` into Exorcist task
structures and can initialize an Exorcist task database from that graph.
"""

from pathlib import Path

import exorcist
import networkx as nx
from gufe import AlchemicalNetwork

from openfe.storage.warehouse import WarehouseBaseClass


def alchemical_network_to_task_graph(
    alchemical_network: AlchemicalNetwork, warehouse: WarehouseBaseClass
) -> nx.DiGraph:
    """Build a global task DAG from an alchemical network.

    Parameters
    ----------
    alchemical_network : AlchemicalNetwork
        Network containing transformations to execute.
    warehouse : WarehouseBaseClass
        Warehouse used to persist protocol units as tasks while the graph is
        constructed.

    Returns
    -------
    nx.DiGraph
        A directed acyclic graph where each node is a task ID in the form
        ``"<transformation_key>:<protocol_unit_key>"`` and edges encode
        protocol-unit dependencies.

    Raises
    ------
    ValueError
        Raised if the assembled task graph is not acyclic.
    """

    global_dag = nx.DiGraph()
    for transformation in alchemical_network.edges:
        dag = transformation.create()
        for unit in dag.protocol_units:
            node_id = f"{str(transformation.key)}:{str(unit.key)}"
            global_dag.add_node(
                node_id,
            )
            warehouse.store_task(unit)
        for u, v in dag.graph.edges:
            u_id = f"{str(transformation.key)}:{str(u.key)}"
            v_id = f"{str(transformation.key)}:{str(v.key)}"
            global_dag.add_edge(u_id, v_id)

    if not nx.is_directed_acyclic_graph(global_dag):
        raise ValueError("AlchemicalNetwork produced a task graph that is not a DAG.")

    return global_dag


def build_task_db_from_alchemical_network(
    alchemical_network: AlchemicalNetwork,
    warehouse: WarehouseBaseClass,
    db_path: Path | None = None,
    max_tries: int = 1,
) -> exorcist.TaskStatusDB:
    """Create and populate a task database from an alchemical network.

    Parameters
    ----------
    alchemical_network : AlchemicalNetwork
        Network containing transformations to convert into task records.
    warehouse : WarehouseBaseClass
        Warehouse used to persist protocol units while building the task DAG.
    db_path : pathlib.Path or None, optional
        Location of the SQLite-backed Exorcist database. If ``None``, defaults
        to ``Path("tasks.db")`` in the current working directory.
    max_tries : int, default=1
        Maximum number of retries for each task before Exorcist marks it as
        ``TOO_MANY_RETRIES``.

    Returns
    -------
    exorcist.TaskStatusDB
        Initialized task database populated with graph nodes and dependency
        edges derived from ``alchemical_network``.
    """
    if db_path is None:
        db_path = Path("tasks.db")

    global_dag = alchemical_network_to_task_graph(alchemical_network, warehouse)
    db = exorcist.TaskStatusDB.from_filename(db_path)
    db.add_task_network(global_dag, max_tries)
    return db
