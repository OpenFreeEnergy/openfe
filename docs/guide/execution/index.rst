.. _userguide_execution:

Execution
=========

Once you have planned your calculations, each :class:`.Transformation` needs to be executed.
Its simulations are run and the results analysed to give a free energy estimate.
**openfe** supports two approaches, which differ in who organises the work.
Both can be used from the command line or from Python.

:ref:`Running Transformations individually <userguide_quickrun>`
   You run one Transformation at a time and decide where and when each one runs,
   for example by submitting one cluster job per Transformation.
   On the command line this is ``openfe quickrun``; in Python, :func:`.execute_DAG`.

:ref:`Task-based execution <userguide_task_based_execution>` (experimental)
   **openfe** splits a whole :class:`.AlchemicalNetwork` into tasks and keeps track of which are ready to run.
   You start one or more workers, and each one picks up the next available task:
   ``openfe run-task`` on the command line, or :class:`.Worker` in Python.

.. toctree::
   execution_theory
   quickrun_execution
   task_based_execution

.. admonition:: Related How-To Guides

   * :doc:`/cookbook/dumping_transformation`
   * :doc:`/cookbook/run_quickrun`
   * :doc:`/cookbook/resume_quickrun`
   * :doc:`/cookbook/run_on_slurm`
   * :doc:`/cookbook/run_repeats_in_parallel`
   * :doc:`/cookbook/run_with_mps`
   * :doc:`/cookbook/run_task_based`
