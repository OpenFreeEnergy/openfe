**Added:**

* ``raise_errors`` argument added to ``bulk_assign_partial_charges`` method to allow to skip raising an ``ExceptionGroup`` with the details of any molecules which fail to have partial charges assigned, by default this is set to ``True`` to maintain the previous behavior (`PR #2180 <https://github.com/OpenFreeEnergy/openfe/pull/2180>`_).

**Changed:**

* The ``bulk_assign_partial_charges`` method now raises an ``ExceptionGroup`` with the details of any molecules which fail
  to have partial charges assigned (`PR #2171 <https://github.com/OpenFreeEnergy/openfe/pull/2171>`_).

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
