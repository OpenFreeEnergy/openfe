"""Task orchestration utilities backed by Exorcist and a warehouse."""

from dataclasses import dataclass
from pathlib import Path

from exorcist.taskdb import TaskStatusDB
from gufe.protocols.protocoldag import _pu_to_pur
from gufe.protocols.protocolunit import (
    Context,
    ProtocolUnit,
    ProtocolUnitResult,
)
from gufe.storage.externalresource.base import ExternalStorage
from gufe.storage.externalresource.filestorage import FileStorage
from gufe.tokenization import GufeKey

from openfe.storage.warehouse import FileSystemWarehouse

from .exorcist_utils import (
    alchemical_network_to_task_graph,
    build_task_db_from_alchemical_network,
)


@dataclass
class Worker:
    """Execute protocol units from an Exorcist task database.

    Parameters
    ----------
    warehouse : FileSystemWarehouse
        Warehouse used to load queued tasks and store execution results.
    task_db_path : pathlib.Path, default=Path("./warehouse/tasks.db")
        Path to the Exorcist SQLite task database.
    """

    warehouse: FileSystemWarehouse
    task_db_path: Path = Path("./warehouse/tasks.db")

    _RESULT_INDEX_PREFIX = "protocol_unit_results"
    _TASK_WORKDIR_PREFIX = "task_workdirs"

    @staticmethod
    def _collect_protocol_unit_keys(value: object) -> set[GufeKey]:
        """Collect `ProtocolUnit` keys from nested unit inputs."""

        if isinstance(value, ProtocolUnit):
            return {value.key}

        found: set[GufeKey] = set()
        if isinstance(value, dict):
            items = value.values()
        elif isinstance(value, list):
            items = value
        else:
            return found

        for item in items:
            found.update(Worker._collect_protocol_unit_keys(item))
        return found

    @classmethod
    def _result_index_location(cls, source_key: GufeKey) -> str:
        return f"{cls._RESULT_INDEX_PREFIX}/{source_key}"

    @classmethod
    def _task_workdir_name(cls, taskid: str) -> str:
        return taskid.replace(":", "__")

    def _task_workspace_paths(
        self, taskid: str, scratch_root: Path, shared_root: Path
    ) -> tuple[Path, Path]:
        workdir_name = self._task_workdir_name(taskid)
        task_scratch = scratch_root / self._TASK_WORKDIR_PREFIX / workdir_name
        task_shared = shared_root / self._TASK_WORKDIR_PREFIX / workdir_name
        return task_scratch, task_shared

    def _store_result_index(self, result: ProtocolUnitResult) -> None:
        shared_store: ExternalStorage = self.warehouse.stores["shared"]
        location = self._result_index_location(result.source_key)
        shared_store.store_bytes(location, str(result.key).encode("utf-8"))

    def _load_result_from_index(self, source_key: GufeKey) -> ProtocolUnitResult | None:
        shared_store: ExternalStorage = self.warehouse.stores["shared"]
        location = self._result_index_location(source_key)

        if not shared_store.exists(location):
            return None

        with shared_store.load_stream(location) as stream:
            result_key = stream.read().decode("utf-8").strip()

        loaded = self.warehouse.load_result_tokenizable(GufeKey(result_key))
        if isinstance(loaded, ProtocolUnitResult):
            return loaded

        return None

    def _scan_result_store_for_sources(
        self, source_keys: set[GufeKey]
    ) -> dict[GufeKey, ProtocolUnitResult]:
        found: dict[GufeKey, ProtocolUnitResult] = {}

        for location in self.warehouse.result_store.iter_contents():
            if len(found) == len(source_keys):
                break

            loaded = self.warehouse.load_result_tokenizable(GufeKey(location))
            if not isinstance(loaded, ProtocolUnitResult):
                continue

            source_key = loaded.source_key
            if source_key in source_keys and source_key not in found:
                found[source_key] = loaded

        return found

    def _build_input_result_mapping(self, unit: ProtocolUnit) -> dict[GufeKey, ProtocolUnitResult]:
        required_keys = self._collect_protocol_unit_keys(unit.inputs)
        if not required_keys:
            return {}

        results: dict[GufeKey, ProtocolUnitResult] = {}
        unresolved = set(required_keys)

        for source_key in required_keys:
            loaded = self._load_result_from_index(source_key)
            if loaded is not None:
                results[source_key] = loaded
                unresolved.discard(source_key)

        if unresolved:
            scanned = self._scan_result_store_for_sources(unresolved)
            for source_key, loaded in scanned.items():
                results[source_key] = loaded
                self._store_result_index(loaded)
                unresolved.discard(source_key)

        if unresolved:
            missing_keys = ", ".join(sorted(str(k) for k in unresolved))
            raise RuntimeError(
                "Missing ProtocolUnitResult(s) for dependency key(s): "
                f"{missing_keys}. Ensure upstream tasks completed successfully."
            )

        return results

    def _checkout_task(self) -> tuple[TaskStatusDB, str, ProtocolUnit] | None:
        """Check out one available task and load its protocol unit.

        Returns
        -------
        tuple[TaskStatusDB, str, ProtocolUnit] or None
            The open database connection, checked-out task ID, and corresponding
            protocol unit, or ``None`` if no task is currently available.
            The caller is responsible for calling ``mark_task_completed`` on the
            returned database using the returned task ID.
        """

        db: TaskStatusDB = TaskStatusDB.from_filename(self.task_db_path)
        # The format for the taskid is "Transformation-<HASH>:ProtocolUnit-<HASH>"
        taskid = db.check_out_task()
        if taskid is None:
            return None

        _, protocol_unit_key = taskid.split(":", maxsplit=1)
        unit = self.warehouse.load_task(GufeKey(protocol_unit_key))
        return db, taskid, unit

    def _get_task(self) -> tuple[str, ProtocolUnit]:
        """Return the next available task ID and protocol unit.

        Returns
        -------
        tuple[str, ProtocolUnit]
            The checked-out task ID and corresponding protocol unit.

        Raises
        ------
        RuntimeError
            Raised when no task is available in the task database.
        """

        task = self._checkout_task()
        if task is None:
            raise RuntimeError("No AVAILABLE tasks found in the task database.")
        db, taskid, unit = task
        return taskid, unit

    def execute_unit(self, scratch: Path) -> tuple[str, ProtocolUnitResult] | None:
        """Execute one checked-out protocol unit and persist its result.

        Parameters
        ----------
        scratch : pathlib.Path
            Scratch directory passed to the protocol execution context.

        Returns
        -------
        tuple[str, ProtocolUnitResult] or None
            The task ID and execution result for the processed task, or
            ``None`` if no task is currently available.

        Raises
        ------
        Exception
            Re-raises any exception thrown during protocol unit execution after
            marking the task as failed.
        """

        # 1. Get task/unit
        task = self._checkout_task()
        if task is None:
            return None
        db, taskid, unit = task
        # 2. Construct the context
        # NOTE: On changes to context, this can easily be replaced with external storage objects
        # However, to satisfy the current work, we will use this implementation where we
        # force the use of a FileSystemWarehouse and in turn can assert that an object is FileStorage.
        shared_store = self.warehouse.stores["shared"]
        if not isinstance(shared_store, FileStorage):
            raise TypeError("Expected a FileStorage backend for the shared store")
        shared_root_dir = shared_store.root_dir
        task_scratch, task_shared = self._task_workspace_paths(taskid, scratch, shared_root_dir)
        task_scratch.mkdir(parents=True, exist_ok=True)
        task_shared.mkdir(parents=True, exist_ok=True)
        ctx = Context(task_scratch, shared=task_shared)
        # 3. Execute unit
        try:
            results = self._build_input_result_mapping(unit)
            inputs = _pu_to_pur(unit.inputs, results)
            result = unit.execute(context=ctx, **inputs)
        except Exception:
            db.mark_task_completed(taskid, success=False)
            raise

        db.mark_task_completed(taskid, success=result.ok())
        # 4. output result to warehouse
        # TODO: we may need to end up handling namespacing on the warehouse side for tokenizables
        self.warehouse.store_result_tokenizable(result)
        self._store_result_index(result)
        return taskid, result
