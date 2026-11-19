**Added:**

* <news item>

**Changed:**

* The parent of the scratch directory passed to ``worker.execute_unit(scratch="path/to/scratch/")`` must already exist.
This change avoids unintentional nested directory creation (`PR #2365 <https://github.com/OpenFreeEnergy/openfe/pull/2365>`_).

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Fixed ``worker.execute_unit()`` so that input validation is performed _before_ a task is marked as ``IN_PROGRESS`` (`PR #2355 <https://github.com/OpenFreeEnergy/openfe/pull/2355>`_).

**Security:**

* <news item>
