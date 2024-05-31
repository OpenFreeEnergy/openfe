=========
Changelog
=========

.. current developments

v1.0.1
====================

**Added:**

* Debug script in devtools to test OpenMM installation.
* use rever to manage changelog

**Changed:**

* Updated docs to reference miniforge instead of mambaforge since they are the same now, see https://github.com/conda-forge/miniforge?tab=readme-ov-file#whats-the-difference-between-mambaforge-and-miniforge
* The LomapAtomMapper defaults have now changed to better reflect real-life
  usage. Key kwarg changes include; `max3d=1.0` and `shift=False`

**Fixed:**

* Calling `get_forward_and_reverse_energy_analysis` in the RFE and AFE
  protocols now results a warning if any results are ``None`` due to
  MBAR convergence issues.
* Checkpoint interval default value has been set to 250 ps instead of 1 ps.
  This better matches the previous default for openfe versions < 1.0rc
  (See issue #772).


