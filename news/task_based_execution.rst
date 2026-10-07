**Added:**

* Added Python API functionality for task-based execution, including:
  * ``openfe.setup_task_campaign()``, which creates a task-based campaign from an AlchemicalNetwork (`PR #2174 <https://github.com/OpenFreeEnergy/openfe/pull/2174>`_)
  * ``openfe.Worker``, which is used to execute the task-based campaign (`PR #2172 <https://github.com/OpenFreeEnergy/openfe/pull/2172>`_).
  * ``openfe.storage.FileSystemWarehouse`` and its parent class ``openfe.storage.WarehouseBaseClass`` (`PR #1864 <https://github.com/OpenFreeEnergy/openfe/pull/1864>`_).
  * ``openfe.TaskStatusDB`` and the helper functions ``openfe.get_task_df``, ``openfe.get_dependency_df`` (`PR #2155 <https://github.com/OpenFreeEnergy/openfe/pull/2155>`_).

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* Removed the unused methods ``metadatastore``, ``resultclient``, and ``resultserver`` from ``openfe.storage`` (`PR #1864 <https://github.com/OpenFreeEnergy/openfe/pull/1864>`_).

**Fixed:**

* <news item>

**Security:**

* <news item>
