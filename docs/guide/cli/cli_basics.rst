CLI basics
==========

The ``openfe`` command consists of several subcommands. This is similar to
tools like ``gmx``, which has subcommands like ``gmx mdrun``, or ``conda``,
which has subcommands like ``conda install``.

To get a list of the subcommands and their descriptions, call ``openfe -h``.

.. TODO autogenerate using sphinxcontrib-programoutput

.. command-output:: openfe -h

The ``--log`` option takes a logging configuration file and sets that
logging behavior. If you use it, it must come before the subcommand name.

You can find out more about each subcommand by putting ``--help`` (or ``-h``) *after*
the subcommand name, e.g., ``openfe quickrun --help``, which returns

.. command-output:: openfe quickrun -h


For more details on various commands, see the :ref:`cli-reference`.
