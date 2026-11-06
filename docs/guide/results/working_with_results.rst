.. _userguide_individual_results:

Working with individual results
===============================

With :ref:`execution of your calculations <userguide_execution>` completed,
we can now start looking at what has been produced.
The majority of Protocols will produce estimates of free energy differences between two or more ``ChemicalSystem`` \s
(the current exception being the :class:`.PlainMDProtocol` which just simulates the dynamics of a single system).
Beyond this, the exact data produced by a given Protocol can vary significantly,
for example the :class:`.RelativeHybridTopologyProtocol` protocol will produce graphs to assess the quality of the simulation, alongside trajectory data files.
By comparison, the :class:`.PlainMDProtocol` will only produce the latter.
For exact details on what is produced consult the :ref:`pages for each Protocol<userguide_protocols>`.

.. todo crossref to HREX and MD Protocol docs from issue 743

What a calculation produces
---------------------------

Running a Transformation with :ref:`quickrun <userguide_quickrun>` produces a results JSON file and a work directory.

The **results file** contains the free energy estimate of the Transformation and its uncertainty,
the full :class:`.ProtocolResult` and the output of every task.

The **work directory** contains one subdirectory per task (``ProtocolUnit``) and attempt.
For the Relative Hybrid Topology Protocol, each repeat has a setup, a simulation and an analysis task:

.. code-block:: text

   workdir/
   ├── results.json
   ├── shared_HybridTopologySetupUnit-<key>_attempt_0/                  (hybrid system)
   ├── shared_HybridTopologyMultiStateSimulationUnit-<key>_attempt_0/   (trajectory and checkpoint)
   ├── shared_HybridTopologyMultiStateAnalysisUnit-<key>_attempt_0/     (analysis plots)
   └── ...                                                              (the same for each repeat)

Protocols with several legs in one Transformation, such as the Separated Topologies Protocol,
have these tasks for each leg and repeat, for example ``SepTopComplexSetupUnit`` and ``SepTopSolventSetupUnit``.
The analysis directories contain plots for checking each repeat, such as the MBAR overlap matrix;
see :ref:`multistate_analysis` for how to read them.

How results are structured
--------------------------

Results are built up in three levels, following the :ref:`execution model <userguide_execution_theory>`:

* A :class:`.ProtocolUnitResult` holds the output of a single task, or its error if the task failed.
* A :class:`.ProtocolDAGResult` holds the results of all tasks from one execution of a Transformation,
  including all of its repeats.
* A :class:`.ProtocolResult` combines one or more ``ProtocolDAGResult``\s and computes the free energy estimate
  and its uncertainty, available through its ``get_estimate()`` and ``get_uncertainty()`` methods.

How the estimate and uncertainty are computed depends on the Protocol.
For the Relative Hybrid Topology Protocol, the estimate is the mean of the repeat estimates,
and the uncertainty is their standard deviation.

.. admonition:: How-to guides

   * :doc:`/how_to_guides/load_results`
   * :doc:`/how_to_guides/jq_inspection`

See Also
--------

For how to deal with multiple results forming a network consult the :ref:`working with networks<userguide_result_networks>`
page.
