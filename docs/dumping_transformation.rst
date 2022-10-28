Dumping a ``Transformation`` to JSON
====================================

If you're trying to run a full campaign of simulations representing an
alchemical network, we generally recommend saving objects using our storage
tools, when avoids saving duplicate information to disk.

.. TODO: add links to storage tools once they're complete

However, there are situations where it is reasonable to serialize a single
:class:`.Transformation`. For example, this can be useful when trying to
compare results run on different machines. This also provides a trivial way
for a user to run edges in parallel, if they don't want to use the more
sophisticated techniques we have developed.

For these cases, we have made it very easy for a user to dump a
transformation to JSON. Simply use the method
:meth:`.Transformation.dump`. This can take a filename (pathlike) or an
already-opened file object. For example:

.. code::

    transformation.dump("mytransformation.json")

Be aware this this is not designed to be space-efficient. That is, if you
have the same object in memory used in two location (for example, the same
``ProteinComponent`` in both ``ChemicalSystems``, then 

This can be reloaded into memory with the :meth:`.Transformation.load`
method:

.. code::

    transformation = Transformation.load("mytransformation.json")

Once you've saved to it JSON, you can also run this transformation with the
``openfe`` command line tool's :ref:`cli_quickrun`, e.g.:

.. code:: bash

    $ openfe quickrun mytransformation.json -d dir_for_files -o output.json
