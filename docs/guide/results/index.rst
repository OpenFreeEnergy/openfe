.. _userguide_results:

Working with Results
====================

With :ref:`execution of your calculations <userguide_execution>` completed,
we can now start looking at what has been produced.
At the very least, Protocols will produce estimates of free energy differences of the ``ChemicalSystem`` \s
(with the exception of MD Protocols which just simulate the behaviour of a single system).
Beyond this, the exact data produced by a given Protocol can vary significantly,
for example the HREX protocol will produce graphs to assess the quality of the simulation,
while the MD Protocol will produce a trajectory file for the simulated system.
For exact details on what is produced consult the :ref:`pages for each Protocol<userguide_protocols>`.

.. todo crossref to HREX and MD Protocol docs from issue 743

How you can inspect these results depends on whether you are doing this from the Python or command line interface.

From a Python interface
-----------------------

Executing a :class:`.ProtocolDAG` using :func:`openfe.execute_DAG` will produce a :class:`.ProtocolDAGResult`,
representing a single iteration of estimating the free energy difference.
One or more of these can be put into the ``.gather()`` method of the ``Protocol`` to form a :class:`.ProtocolResult`,
this class takes care of the averaging and concatenation of different iterations of the estimation process.
This ``ProtocolResult`` class has ``.get_estimate()`` and ``.get_uncertainty()`` methods which return the estimates
of free energy difference along with its uncertainty.

From the command line
---------------------

If you had executed your calculation using the :ref:`quickrun <cli_quickrun>` command,
then a ``.json`` results log file as well as a directory of files will have been produced.
Most importantly, the ``.json`` results file has ``estimate`` and ``uncertainty`` keys,
which serve the same purpose as the ``get_estimate()`` and ``get_uncertainty()`` methods described above.

The :ref:`openfe gather <cli_gather>` command offers a way to collate information across many different individual
simulations and prepare a table of results.
