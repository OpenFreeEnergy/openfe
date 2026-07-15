**Added:**

* <news item>

**Changed:**

* Small molecules in RelativeHybridTopologyProtocol topologies (including the output PDB) are now named LG1 amd LG2 (alchemical ligand) and COF (cofactors) instead of UNK. If a residue name was already assigned, the assigned one is kept.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Fixed inflated ligand RMSD in the RelativeHybridTopology protocol's structural analysis for systems containing cofactors; the ligand RMSD is now computed for the alchemical ligand alone rather than conflating it with cofactors that shared the UNK residue name.

**Security:**

* <news item>
