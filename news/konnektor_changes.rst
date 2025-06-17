**Added:**

* Added optional ``n_processes`` (number of parallel processes to use when generating the network) arguments for network planners.
* Added optional ``progress`` (whether to show progress bar) for ``openfe.setup.ligand_network_planning.generate_radial_network`` (default=``False``, such that there is no default behavior change).
  
**Changed:**

* `konnektor <https://konnektor.openfree.energy/en/latest/>_` is now used as the backend for all network generation.
* ``openfe.setup.ligand_network_planning.generate_maximal_network`` now returns the *best* mapping for each edge, rather than *all possible* mappings for each edge. If multiple mappers are passed but no scorer, the first mapper passed will be used, and a warning will be raised.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
