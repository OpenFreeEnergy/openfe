"""Utilities for building Exorcist task graphs and task databases."""

from pathlib import Path

import exorcist
import networkx as nx
from gufe import AlchemicalNetwork

from openfe.storage.warehouse import WarehouseBaseClass


def alchemical_network_to_task_graph(
    alchemical_network: AlchemicalNetwork, warehouse: WarehouseBaseClass
) -> nx.DiGraph:
    """Build a global task DAG from an AlchemicalNetwork."""

    global_dag = nx.DiGraph()
    for transformation in alchemical_network.edges:
        dag = transformation.create()
        for unit in dag.protocol_units:
            node_id = f"{transformation.name}-{transformation.key}:{unit.name}-{unit.key}"
            global_dag.add_node(
                node_id,
                label=f"{transformation.name}\n{unit.name}",
                transformation_key=str(transformation.key),
                protocol_unit_key=str(unit.key),
            )
            warehouse.store_task(unit)
        for u, v in dag.graph.edges:
            u_id = f"{transformation.key}:{u.key}"
            v_id = f"{transformation.key}:{v.key}"
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
    """Create an Exorcist TaskStatusDB from an AlchemicalNetwork."""
    if db_path is None:
        db_path = Path("tasks.db")

    global_dag = alchemical_network_to_task_graph(alchemical_network, warehouse)
    db = exorcist.TaskStatusDB.from_filename(db_path)
    db.add_task_network(global_dag, max_tries)
    return db
