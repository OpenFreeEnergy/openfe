from dataclasses import dataclass
from pathlib import Path

from exorcist.taskdb import TaskStatusDB
from gufe.protocols.protocoldag import _pu_to_pur
from gufe.protocols.protocolunit import (
    Context,
    ProtocolUnit,
    ProtocolUnitFailure,
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

    def _get_task(self) -> ProtocolUnit:
        # Right now, we are just going to assume it exists in the warehouse folder
        location = Path("./warehouse/tasks.db")

        db: TaskStatusDB = TaskStatusDB.from_filename(location)
        # The format for the taskid is going to "Transformation-<HASH>:Unit<HASH>"
        taskid = db.check_out_task()
        # Load the unit from warehouse and return
        _, protocol_unit_key = taskid.split(":", maxsplit=1)

        return self.warehouse.load_task(GufeKey(protocol_unit_key))

    def execute_unit(self, scratch: Path):
        # 1. Get task/unit
        unit = self._get_task()
        # 2. Constrcut the context
        # NOTE: On changes to context, this can easily be replaced with external storage objects
        # However, to satisfy the current work, we will use this implementation where we
        # force the use of a FileSystemWarehouse and in turn can assert that an object is FileStorage.
        shared_store: FileStorage = self.warehouse.stores["shared"]
        shared_root_dir = shared_store.root_dir
        ctx = Context(scratch, shared=shared_root_dir)
        results: dict[GufeKey, ProtocolUnitResult] = {}
        inputs = _pu_to_pur(unit.inputs, results)
        # 3. Execute unit
        result = unit.execute(context=ctx, **inputs)
        # if not result.ok():
        # Increment attempt in taskdb
        # 4. output result to warehouse
        # TODO: we may need to end up handling namespacing on the warehouse side for tokenizables
        self.warehouse.store_result_tokenizable(result)
