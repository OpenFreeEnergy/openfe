=========
Changelog
=========

.. current developments

v1.2.0
====================

**Added:**

* New `cookbook featuring bespokefit <https://docs.openfree.energy/en/stable/cookbook/bespoke_parameters.html>`_

**Fixed:**

* Improved responsiveness of CLI calls
* Fixed bug where `openfe gather --report raw` was only including first replicates.



v1.1.0
====================

**Added:**

* Extended system solvation tooling, including support for; non-cubic boxes,
  explicitly defining the number of waters added, the box vectors, and box size
  as supported by `Modeller.addSolvent` in OpenMM 8.0 and above.

**Changed:**

* Improved documentation of the OpenMMSolvationSettings.
* The `PersesAtomMapper` now uses openff.units inline with the rest of the package.
* Structural analysis data is no longer written to `structural_analysis.json`
  but rather a 32bit numpy compressed file named `structural_analysis.npz`
  (PR #937).
* Structural analysis array data is no longer directly returned in the
  RelativeHybridTopologyProtocol result dictionary. Instead it should
  be accessed from the serialized NPZ file `structural_analysis.npz`.
  The `structural_analysis` key now contains a path to the NPZ file,
  if the structural analysis did not fail (the `structural_analysis_error`
  key will instead be present on failure) (PR #937).
* Add duecredit citations for pymbar when calling
  `openfe.protocols.openmm_utils.multistate_analysis`.

**Fixed:**

* 2D RMSD plotting now allows for fewer than 5 states (PR #896).
* 2D RMSD plotting no longer draws empty axes when
  the number of states - 1 is not divisible by 4 (PR #896).
* The RelativeHybridTopologyProtocol result unit is now much smaller,
  due to the removal of structural analysis data (PR #937).



v1.0.1
====================

**Added:**

* Debug script in devtools to test OpenMM installation.
* Use rever to manage changelog.

**Changed:**

* Updated docs to reference miniforge instead of mambaforge since they are the same now, see https://github.com/conda-forge/miniforge?tab=readme-ov-file#whats-the-difference-between-mambaforge-and-miniforge.
* The LomapAtomMapper defaults have now changed to better reflect real-life usage. Key kwarg changes include; `max3d=1.0` and `shift=False`.

**Fixed:**

* Calling `get_forward_and_reverse_energy_analysis` in the RFE and AFE protocols now results a warning if any results are ``None`` due to MBAR convergence issues.
* Checkpoint interval default value has been set to 250 ps instead of 1 ps.
  This better matches the previous default for openfe versions < 1.0rc
  (See issue #772).


