**Added:**

* The ``AbsoluteBindingProtocol`` and ``AbsoluteSolvationProtocol`` now
  carry out structural analyses at the end of the simulation. For all
  legs, except vacuum, a symmetry-corrected ligand RMSD is calculated. For
  complex transformations, a ligand COM drift and protein 2D RMSD is also
  calculated. All results are written to a numpy NPZ file named
  `structural_analysis.npz`, alongside PNGs for the plots for each
  analysis type.
* A new ``analysis_settings`` field has been added to
  ``AbsoluteBindingSettings`` and ``AbsoluteSolvationSettings``
  to control post-simulation analysis.

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* <news item>

**Security:**

* <news item>
