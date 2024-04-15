Relative Hybrid Topology Protocol
=================================

Overview
--------

The relative free energy calculation approach calculates the difference in 
free energy between two similar ligands. Depending on the :class:`.ChemicalSystem` 
provided, the protocol either calculates the relative binding free energy 
(RBFE), or the relative hydration free energy (RHFE). 
.. todo: Add reference to ChemicalSystem section

In a thermodynamic 
cycle, one ligand is converted into the other ligand by alchemically 
transforming the atoms that vary between the two ligands. The 
transformation is carried out in both environments, meaning both in the 
solvent (ΔG\ :sub:`solv`\) and in the binding site (ΔG\ :sub:`site`\) for RBFE calculations 
and in the solvent (ΔG\ :sub:`solv`\) and vacuum (ΔG\ :sub:`vacuum`\) for RHFE calculations.

.. _label: Thermodynamic cycle for the relative binding free energy protocol
.. figure:: img/rbfe_thermocycle.png
   :scale: 50%

   Thermodynamic cycle for the relative binding free energy protocol.
   
Scientific Details
------------------

This :class:`.RelativeHybridTopologyProtocol` is based off the `Perses implementation <https://perses.readthedocs.io/en/latest/>`_.

The Hybrid Topology approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.RelativeHybridTopologyProtocol` uses a hybrid topology approach to represent the two
ligands, meaning that a single set of coordinates is used to represent the
common core of the two ligands while the atoms that differ between the two
ligands are represented separately. An atom map defines which atoms belong
to the core (mapped atoms) and which atoms are unmapped and represented
separately (see :ref:`Creating atom mappings <Creating Atom Mappings>`). During the alchemical transformation, mapped atoms are switched
from the type in one ligand to the type in the other ligands, while unmapped
atoms are switched on or off, depending on which ligand they belong to.

.. note:: In this hybrid topology approach, all bonded interactions between the dummy region and the core region are kept. 
          As pointed out by Fleck et al. [1]_, this can lead to systematic errors if the contribution of the dummy group does not cancel out
          in the thermodynamic cycle (no separability of the partition function). We are currently working on fixing this issue.

The lambda schedule
~~~~~~~~~~~~~~~~~~~

The protocol interpolates molecular interactions between the initial and final state of the perturbation using a discrete set of lambda windows. A function describes how the different lambda components (bonded and nonbonded terms) are interpolated.
Only parameters that differ between state A (``lambda=0``) and state B (``lambda=1``) are interpolated. 
In the default lambda function in the :class:`.RelativeHybridTopologyProtocol`, first the electrostatic interactions of state A are turned off while simultaneously turning on the steric interactions of state B. Then, the steric interactions of state A are turned off while simultaneously turning on the electrostatic interactions of state B. Bonded interactions are interpolated linearly between ``lambda=0`` and ``lambda=1``. The ``lambda_settings`` ``lambda_functions`` and ``lambda_windows`` define the alchemical pathway.
A soft-core potential is applied to the Lennard-Jones potential to avoid instablilites in intermediate lambda windows.
Both the soft-core potential functions from Beutler et al. [2]_ and from Gapsys et al. [3]_ are available and can be specified in the ``alchemical_settings.softcore_LJ`` settings
(default: ``gapsys``).

Simulation overview
~~~~~~~~~~~~~~~~~~~

The :class:`.ProtocolDAG` of the :class:`.RelativeHybridTopologyProtocol` contains the :class:`.ProtocolUnit`\ s from one leg of the thermodynamic
cycle. 
This means that each :class:`.ProtocolDAG` only runs a single leg of a thermodynamic cycle and therefore two Protocol instances need to be run to get the overall relative free energy difference, ΔΔG. 
If multiple ``protocol_repeats`` are run (default: ``protocol_repeats=3``), the :class:`.ProtocolDAG` contains multiple :class:`.ProtocolUnit`\ s of both vacuum and solvent transformations.

Simulation Steps
""""""""""""""""

Each :class:`.ProtocolUnit` carries out the following steps:

1. Parameterize the system using `OpenMMForceFields <https://github.com/openmm/openmmforcefields>`_ and `Open Force Field <https://github.com/openforcefield/openff-forcefields>`_.
2. Create an alchemical system (hybrid topology).
3. Minimize the alchemical system.
4. Equilibrate and production simulate the alchemical system using the chosen multistate sampling method (under NPT conditions if solvent is present).
5. Analyze results for the transformation (for a single leg in the thermodynamic cycle).

Note: three different types of multistate sampling (i.e. replica swapping between lambda states) methods can be chosen; HREX, SAMS, and independent (no lambda swaps attempted). By default the HREX approach is selected, this can be altered using ``simulation_settings.sampler_method`` (default: ``repex``).

Simulation details
""""""""""""""""""

Here are some details of how the simulation is carried out which are not detailed in the :class:`.RelativeHybridTopologySettings`:

* The protocol applies a `LangevinMiddleIntegrator <https://openmmtools.readthedocs.io/en/latest/api/generated/openmmtools.mcmc.LangevinDynamicsMove.html>`_ which uses Langevin dynamics, with the LFMiddle discretization [4]_.
* A MonteCarloBarostat is used in the NPT ensemble to maintain constant pressure.

Getting the free energy estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The free energy differences are obtained from simulation data using the `MBAR estimator <https://www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio>`_ (multistate Bennett acceptance ratio estimator)
as implemented in the `PyMBAR package <https://pymbar.readthedocs.io/en/master/mbar.html>`_.
In addition to the MBAR estimates of the two legs of the thermodynamic cycle and the overall relative binding free energy difference,
the protocol also returns some metrics to help assess convergence of the results,
these are detailed in the :ref:`multistate analysis section <multistate_analysis>`.

.. todo: issue 792, consolidate this page into its own analysis page and link both RBFE and AFE pages to it
.. _multistate_analysis:

Analysis
~~~~~~~~

As standard, some analysis of the each simulation repeat is performed.
This analysis is made available through either the dictionary of results in the execution output,
or through some ready-made plots for quick inspection.
This analysis can be categorised as relating
to the energetics of the different lambda states that were sampled,
or to the analysis of the change in structural conformation over time in each state.

Energetic and replica exchange analysis
"""""""""""""""""""""""""""""""""""""""

These analyses consider the swapping and energetic overlap between the
different simulated states to help assess the convergence and correctness of the estimate of free energy
difference produced.

.. list-table:: Energetic Analysis examples
  :widths: 75 25
  :header-rows: 1

  * - Description
    - Example
  * - **MBAR overlap matrix.**

      This plot is used to assess if the different lambda states simulated overlapped energetically.
      Each matrix element represents the probability of a sample from a given row state being observable in a given column
      state.
      Since the accuracy of the MBAR estimator depends on sufficient overlap between lambda states, this is a very
      important metric.
      This plot should show that the diagonal of the matrix has some "width" so that the two end states are connected,
      with elements adjacent to the diagonal being at least 0.03 [5]_.
    - .. image:: img/mbar_overlap_matrix.png
  * - **Replica exchange probability matrix** (for replica exchange sampler simulations only).

      Similar to the MBAR overlap matrix, this shows the probability of a given lambda state being exchanged with another.
      Again, the diagonal of this matrix should be at least tridiagonal wide for the two end states to be connected.
    - .. image:: img/replica_exchange_matrix.png
  * - **Forward and reverse convergence of free energy estimates.**

      Using increasingly larger portions of the total data,
      this analysis calculates the free energy difference, both in forward and backward directions.
      In this analysis, forward and backward estimates that agree within error using only a fraction of the total data
      suggest convergence [5]_.
    - .. image:: img/forward_reverse_convergence.png
  * - **Timeseries of replica states.**

      This plot shows the time evolution of the different system configurations as they are
      exchanged between different lambda states.
      This plot should show that the states are freely mixing and that there are no cliques forming.
    - .. image:: img/replica_state_timeseries.png

Structural analysis
"""""""""""""""""""

If a protein was present, these analyses first center and align the system so that
the protein is considered the frame of reference.
Further analysis can be performed by inspecting the ``simulation.nc`` and ``hybrid_system.pdb`` files,
which contain a multistate trajectory and topology for the hybrid system respectively.
These files can be loaded into an MDAnalysis Universe object using the `openfe_analysis`_ package.

.. list-table:: Structural Analysis examples
  :widths: 75 25
  :header-rows: 1

  * - Description
    - Example
  * - **Ligand RMSD.**

      This produces a plot called ``ligand_RMSD.png`` and a results entry ``ligand_RMSD`` which gives the
      RMSD of the ligand molecule over time relative to the first frame of the production phase, for each simulated state.
      Large RMSD values, e.g. greater than 5 angstrom (system dependent), would indicate an unstable ligand binding mode.
    - .. image:: img/ligand_RMSD.png
  * - **Ligand COM drift.**

      For simulations with a protein present, this metric gives the total distance of the ligand COM
      from its initial starting (docked) position.  If this metric increases over the course of the simulation (beyond 5
      angstrom) it indicates that the ligand drifted from the binding pocket, and the simulation is unreliable.
      This produces a plot called ``ligand_COM_drift.png`` and a results entry ``ligand_COM_drift``.
    - .. image:: img/ligand_COM_drift.png
  * - **Protein 2D RMSD.**

      For simulations with a protein present, this metric gives, for each lambda state, the RMSD of the
      protein structure over time, using each frame analysed as a reference frame, to produce a 2 dimensional heatmap.
      This plot should show no significant spikes in RMSD (which will appear as brightly coloured areas).
    - .. image:: img/protein_2D_RMSD.png


See Also
--------

**Setting up RFE calculations**

* :ref:`Setting up RBFE calculations <define-rbfe>`
* :ref:`Setting up RHFE calculations <define-rsfe>`

**Tutorials**

* :any:`Relative Free Energies with the OpenFE CLI <../../tutorials/rbfe_cli_tutorial>`
* :any:`Relative Free Energies with the OpenFE Python API <../../tutorials/rbfe_python_tutorial>`

**Cookbooks**

:ref:`Cookbooks <cookbooks>`

**API Documentation**

* :ref:`OpenMM Relative Hybrid Topology Protocol <rfe protocol api>`
* :ref:`OpenMM Protocol Settings <openmm protocol settings api>`

References
----------
* `pymbar <https://pymbar.readthedocs.io/en/stable/>`_
* `perses <https://perses.readthedocs.io/en/latest/>`_
* `OpenMMTools <https://openmmtools.readthedocs.io/en/stable/>`_
* `OpenMM <https://openmm.org/>`_

.. [1] Dummy Atoms in Alchemical Free Energy Calculations, Markus Fleck, Marcus Wieder, and Stefan Boresch, J. Chem. Theory Comput.2021, 17, 4403−4419
.. [2] Avoiding singularities and numerical instabilities in free energy calculations based on molecular simulations, T.C. Beutler, A.E. Mark, R.C. van Schaik, P.R. Greber, and W.F. van Gunsteren, Chem. Phys. Lett., 222 529–539 (1994)
.. [3] New Soft-Core Potential Function for Molecular Dynamics Based Alchemical Free Energy Calculations, V. Gapsys, D. Seeliger, and B.L. de Groot, J. Chem. Theor. Comput., 8 2373-2382 (2012)
.. [4] Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics, Zhijun Zhang, Xinzijian Liu, Kangyu Yan, Mark E. Tuckerman, and Jian Liu, J. Phys. Chem. A 2019, 123, 28, 6056-6079
.. [5] Guidelines for the analysis of free energy calculations, Pavel V. Klimovich, Michael R. Shirts, and David L. Mobley, J Comput Aided Mol Des. 2015 May; 29(5):397-411. doi: 10.1007/s10822-015-9840-9

.. _openfe_analysis: https://github.com/OpenFreeEnergy/openfe_analysis
