**Added:**

* Added the `_adaptive_settings` class method to the `RelativeHybridTopologyProtocol` which adapts the recommended default settings based on the input transformation type (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

**Changed:**

* The default `time_per_iteration` setting of the `MultiStateSimulationSettings` class has been increased from 1.0 ps to 2.5 ps as part of the fast settings update (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

* The default `box_shape` setting of the `OpenMMSolvationSettings` class has been changed from `cubic` to `dodecahedron` to improve simulation efficiency as part of the fast settings update (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

* The default `solvent_padding` settings of the `OpenMMSolvationSettings` class has been increased from 1.2 nm to 1.5 nm to be compatible with the new `box_shape` default as part of the fast settings update (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

* When calling the CLI `openfe plan_rbfe_network`, the `RelativeHybridTopologyProtocol` settings now reflects the above "fast" settings updates. This includes;
  * Dodecahedron box solvation
  * Solvation cutoff of 1.5 nm in solvent-only legs, and 1.0 nm in complex legs
  * A replica exchange rate of 2.5 ps
  * A 0.9 nm nonbonded cutoff

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
