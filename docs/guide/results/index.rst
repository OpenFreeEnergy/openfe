.. _userguide_results:

Working with Results
====================

With :ref:`execution of your calculations <userguide_execution>` completed,
we can now start looking at what has been produced.
The majority of Protocols will produce estimates of free energy differences between two or more ``ChemicalSystem`` \s
(the current exception being the :class:`.PlainMDProtocol` which just simulates the dynamics of a single system).
Beyond this, the exact data produced by a given Protocol can vary significantly,
for example the :class:`.RelativeHybridTopologyProtocol` protocol will produce graphs to assess the quality of the simulation, alongside trajectory data files.
By comparison, the :class:`.PlainMDProtocol` will only produce the latter.
For exact details on what is produced consult the :ref:`pages for each Protocol<userguide_protocols>`.

.. todo crossref to HREX and MD Protocol docs from issue 743

How you can inspect these results depends on whether you have executed your simulations
from the command line or a Python script.

From command line execution
---------------------------

If you had executed your calculation using the :ref:`quickrun <cli_quickrun>` command,
then a ``.json`` results log file as well as a directory of files will have been produced.
This directory will have various plots and results of analysis, the exact details of which are described
in the :ref:`pages for each Protocol<userguide_protocols>`.

Most importantly, the ``.json`` results file has ``estimate`` and ``uncertainty`` keys,
which serve the same purpose as the ``get_estimate()`` and ``get_uncertainty()`` methods described above.
The full ``json`` results file can be reloaded into a Python session as::

  >>> import gufe
  >>> import json
  >>>
  >>> with open('././Transformation-97d7223f918bbdb0570edc2a49bbc43e_results.json', 'r') as f:
  ...     results = json.load(f, cls=gufe.tokenization.JSON_HANDLER.decoder)
  >>> results['estimate']
  -19.889719513104342 <Unit('kilocalorie_per_mole')>
  >>> results['uncertainty']
  0.574685524681712 <Unit('kilocalorie_per_mole')>



From Python execution
---------------------

Executing a :class:`.ProtocolDAG` using :func:`openfe.execute_DAG` will produce a :class:`.ProtocolDAGResult`,
representing a single iteration of estimating the free energy difference.
One or more of these can be put into the ``.gather()`` method of the ``Protocol`` to form a :class:`.ProtocolResult`,
this class takes care of the averaging and concatenation of different iterations of the estimation process.
This ``ProtocolResult`` class has ``.get_estimate()`` and ``.get_uncertainty()`` methods which return the estimates
of free energy difference along with its uncertainty.

See Also
--------

For how to deal with multiple results forming a network consult the :ref:`working with networks<userguide_result_networks>`
page.
