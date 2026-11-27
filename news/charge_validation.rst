**Added:**

* An option to assign run time partial charges using a small molecule force field. Set the ``partial_charge_method`` of  :class:`OpenFFPartialChargeSettings` to ``forcefield`` in the protocol settings (`PR #2132 <https://github.com/OpenFreeEnergy/openfe/pull/2132>`_).

**Changed:**

* All protocols will raise a ``ProtocolValidationError`` error if a non-deterministic partial charge generation method (AM1-BCC or AM1-BCC-ELF10) is used at run time, please provide molecules with pre-calculated charges to use ``am1bcc`` or use a deterministic method such as library charges or AshGC (`PR #2132 <https://github.com/OpenFreeEnergy/openfe/pull/2132>`_).

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
