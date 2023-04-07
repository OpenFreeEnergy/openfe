
Execution
---------

Given a :class:`.Transformation`, the easiest way to run it is to use the
:func:`.execute_DAG` method. This will take the ... and return a
:class:`.ProtocolDAGResult`. Multiple ProtocolDAGResults from a given
transformation can be analyzed together with :meth:`.Protocol.gather` to
create a :class:`.ProtocolResult`.




