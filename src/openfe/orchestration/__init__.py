from dataclasses import dataclass
from pathlib import Path

from exorcist.taskdb import TaskStatusDB
from gufe.protocols.protocoldag import _pu_to_pur
from gufe.protocols.protocolunit import (
    Context,
    ProtocolUnit,
    ProtocolUnitResult,
)
from gufe.storage.externalresource.filestorage import FileStorage
from gufe.tokenization import GufeKey

from openfe.storage.warehouse import FileSystemWarehouse

from .exorcist_utils import (
    alchemical_network_to_task_graph,
    build_task_db_from_alchemical_network,
)


@dataclass
class Worker:
    warehouse: FileSystemWarehouse
    task_db_path: Path = Path("./warehouse/tasks.db")

    def _checkout_task(self) -> tuple[str, ProtocolUnit] | None:
        db: TaskStatusDB = TaskStatusDB.from_filename(self.task_db_path)
        # The format for the taskid is "Transformation-<HASH>:ProtocolUnit-<HASH>"
        taskid = db.check_out_task()
        if taskid is None:
            return None

        _, protocol_unit_key = taskid.split(":", maxsplit=1)
        unit = self.warehouse.load_task(GufeKey(protocol_unit_key))
        return taskid, unit

    def _get_task(self) -> ProtocolUnit:
        task = self._checkout_task()
        if task is None:
            raise RuntimeError("No AVAILABLE tasks found in the task database.")
        _, unit = task
        return unit

    def execute_unit(self, scratch: Path) -> tuple[str, ProtocolUnitResult] | None:
        # 1. Get task/unit
        task = self._checkout_task()
        if task is None:
            return None
        taskid, unit = task
        # 2. Constrcut the context
        # NOTE: On changes to context, this can easily be replaced with external storage objects
        # However, to satisfy the current work, we will use this implementation where we
        # force the use of a FileSystemWarehouse and in turn can assert that an object is FileStorage.
        shared_store: FileStorage = self.warehouse.stores["shared"]
        shared_root_dir = shared_store.root_dir
        ctx = Context(scratch, shared=shared_root_dir)
        results: dict[GufeKey, ProtocolUnitResult] = {}
        inputs = _pu_to_pur(unit.inputs, results)
        db: TaskStatusDB = TaskStatusDB.from_filename(self.task_db_path)
        # 3. Execute unit
        try:
            result = unit.execute(context=ctx, **inputs)
        except Exception:
            db.mark_task_completed(taskid, success=False)
            raise

        db.mark_task_completed(taskid, success=result.ok())
        # 4. output result to warehouse
        # TODO: we may need to end up handling namespacing on the warehouse side for tokenizables
        self.warehouse.store_result_tokenizable(result)
        return taskid, result
