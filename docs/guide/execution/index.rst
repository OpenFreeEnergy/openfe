.. _userguide_execution:

Execution
=========

With a :class:`.Transformation` defined, the next step is to execute this.
The easiest way to run it is to use the :ref:`quickrun CLI tool <userguide_quickrun>`.
More advanced options are available through first considering the
:ref:`theory of the execution model<userguide_execution_theory>`
then :ref:`reading on the available Python functions<reference_execution>`.

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
