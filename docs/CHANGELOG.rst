=========
Changelog
=========

.. current developments

v1.3.1
====================
Bugfix release - Improved error handling and code cleanup.

We are also dropping official support for MacOSX-x86_64.
Any platform-specific bugs will be addressed when possible, but as a low priority.

**Added:**

* ``openfe gather`` now detects failed simulations up-front and prints warnings to stdout (PR #1207).

**Changed:**

* Temporarily disabled bootstrap uncertainties in forward/reverse analysis due to solver loop issues when dealing with too small a set of samples (PR #1174).

**Removed:**

* Dropped official support for MacOSX-x86_64. Any platform-specific bugs will be addressed when possible, but as a low priority.
* Unused trajectory handling code was removed from ``openfe.utils``, please use ``openfe-analysis`` instead (PR #1182).

**Fixed:**

* Fixed issue #1178 -- The GPU system probe is now more robust to different ways the ``nvidia-smi`` command can fail (PR #1186)
* Fixed bug where openmm protocols using default settings would re-load from JSON as a different gufe key due to unit name string representation discrepancies (PR #1210)


v1.3.0
====================

**Added:**

* Added CLI support for ``generate_lomap_network``. This option can be specified as a `YAML-defined setting <https://docs.openfree.energy/en/stable/guide/cli/cli_yaml.html>`_
* Added ``--n-protocol-repeats`` CLI option to allow user-defined number of repeats per quickrun execution. This allows for parallelizing execution of repeats by setting ``--n-protocol-repeats=1`` and calling ``quickrun`` on the same input file multiple times.
* Added a new CLI command (``charge-molecules``) to bulk assign partial charges to molecules `PR#1068 <https://github.com/OpenFreeEnergy/openfe/pull/1068>`_
* CLI setup will raise warnings for unsupported top-level YAML fields.
* OpenMMEngineSettings now has a `gpu_device_index` attribute allowing users to pass through a list of ``ints`` to select the GPU devices to run their simulations on.
* Add support for variable position/velocity trajectory writing.
* ``openfe gather`` now supports replicates that have been submitted in parallel across separate directories.

**Changed:**

* Networks planned using the CLI will now automatically use an extended protocol for transformations involving a net charge change `PR#1053 <https://github.com/OpenFreeEnergy/openfe/pull/1053>`_
* The ``plan-rhfe-network`` and ``plan-rbfe-network`` CLI commands will now assign partial charges before planning the network if charges are not present, the charge assignment method can be controlled via the yaml settings file `PR#1068 <https://github.com/OpenFreeEnergy/openfe/pull/1068>`_
* `openfe.protocols.openmm_rfe._rfe_utils.compute` has been moved to `openfe.protocols.openmm_utils.omm_compute`.
* ``openfe gather`` now includes *all* edges with missing runs (instead of just the first failing edge) when raising a "missing runs" error.
* ``openfe quickrun`` now creates the parent directory as-needed for user-defined output json paths (``-o``).
* The MBAR bootstrap (1000 iterations) error is used to estimate protocol uncertainty instead of the statistical uncertainty (one standard deviation) and pymbar3 is no longer supported `PR#1077 <https://github.com/OpenFreeEnergy/openfe/pull/1077>`_
* CLI network planners' default names use prefixes `rbfe_` or `rhfe_` , instead of `easy_rbfe` or `easy_rhfe`, to simplify default transformation names.

**Removed:**

* openfe is no longer tested against macos-12. macos support is, for now, limited to osx-arm64 (macos-14+).

**Fixed:**

* ``openfe quickrun`` now creates the parent directory as-needed for user-defined output json paths (``-o``).
* OpenMM CPU vacuum calculations now enforce the use of a single CPU to avoid large performance losses.



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


