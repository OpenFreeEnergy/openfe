.. _userguide_execution:

Execution
---------

Given a :class:`.Transformation`, the easiest way to run it is to use the
:func:`.execute_DAG` method. This will take the `.Transformation` object
and execute its `.ProtocolUnit` instances serially. Once complete it will
return a  :class:`.ProtocolDAGResult`. Multiple ProtocolDAGResults from a given
transformation can be analyzed together with :meth:`.Protocol.gather` to
create a :class:`.ProtocolResult`.


.. TODO: add information about failures etc...

.. toctree::
   execution