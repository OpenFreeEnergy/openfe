**Added:**

* Added the `_adaptive_settings` class method to the `RelativeHybridTopologyProtocol` which adapts the recommended default settings based on the input transformation type (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

**Changed:**

* The default `time_per_iteration` setting of the `MultiStateSimulationSettings` class has been increased from 1.0 ps to 2.5 ps as part of the fast settings update (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

* The default `box_shape` setting of the `OpenMMSolvationSettings` class has been changed from `cubic` to `dodecahedron` to improve simulation efficiency as part of the fast settings update (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

* The default `solvent_padding` settings of the `OpenMMSolvationSettings` class has been increased from 1.2 nm to 1.5 nm to be compatible with the new `box_shape` default as part of the fast settings update (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

* The default settings applied to the `RelativeHybridTopologyProtocol` by the CLI have been updated to reflect the fast settings update, including changes to `time_per_iteration`, `box_shape`, and `solvent_padding` (PR `#1523 <https://github.com/OpenFreeEnergy/openfe/pull/1523>`_).

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
