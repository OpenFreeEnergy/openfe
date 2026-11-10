.. _howto_load_results:

Load results in Python
======================

The results file written by ``openfe quickrun`` can be loaded in Python
to access the free energy estimate as well as the per-repeat estimates and analysis data.

Load the results file
---------------------

Load the file with gufe's JSON decoder, which restores the units of the stored values::

  >>> import gufe
  >>> import json
  >>>
  >>> with open('Transformation-97d7223f918bbdb0570edc2a49bbc43e_results.json', 'r') as f:
  ...     results = json.load(f, cls=gufe.tokenization.JSON_HANDLER.decoder)
  >>> results['estimate']
  <Quantity(-19.8897195, 'kilocalorie_per_mole')>
  >>> results['uncertainty']
  <Quantity(0.574685525, 'kilocalorie_per_mole')>

The dictionary contains four keys:

``estimate`` and ``uncertainty``
   The free energy estimate of the Transformation and its uncertainty.
   If the run failed, both are ``None``.

``protocol_result``
   The full :class:`.ProtocolResult`, including the per-repeat estimates and the analysis data.

``unit_results``
   The outputs, or the error and traceback, of every task (``ProtocolUnit``).
   See :ref:`howto_recover_quickrun` for how to read the error of a failed task.

Recreate the ProtocolResult
---------------------------

To access more than the overall estimate, recreate the :class:`.ProtocolResult`:

.. code-block:: python

   import openfe

   protocol_result = openfe.ProtocolResult.from_dict(results["protocol_result"])

   protocol_result.get_estimate()
   protocol_result.get_uncertainty()
   protocol_result.get_individual_estimates()  # estimate and MBAR uncertainty of each repeat

This returns the result class of the Protocol that was run, for example a :class:`.RelativeHybridTopologyProtocolResult`.

If you ran the calculation in Python
------------------------------------

If you executed the Transformation in Python with :func:`openfe.execute_DAG`,
turn the returned :class:`.ProtocolDAGResult` into a :class:`.ProtocolResult` with the Transformation's
``gather`` method:

.. code-block:: python

   dag_result = openfe.execute_DAG(transformation.create(), shared_basedir=shared, scratch_basedir=scratch)
   protocol_result = transformation.gather([dag_result])

``gather`` takes a list, so that the results of several executions of the same Transformation can be combined.

See also
--------

* :doc:`gather_results` – combining the results of many Transformations into a table.
* :doc:`jq_inspection` – inspecting the results file from the command line.
* :ref:`userguide_individual_results` – what a calculation produces.
