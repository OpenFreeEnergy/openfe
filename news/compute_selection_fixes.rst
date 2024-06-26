**Added:**

* OpenMMEngineSettings now has a `gpu_device_index` attribute
  allowing users to pass through a list of ints to select the
  GPU devices to run their simulations on.

**Changed:**

* `openfe.protocols.openmm_rfe._rfe_utils.compute` has been moved
  to `openfe.protocols.openmm_utils.omm_compute`.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* OpenMM CPU vacuum calculations now enforce the use of a single
  CPU to avoid large performance losses.

**Security:**

* <news item>
