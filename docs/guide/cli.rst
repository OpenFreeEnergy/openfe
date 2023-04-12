Using the CLI
=============

In addition to the powerful Python API, OpenFE provides a simple command
line interface to facilitate some more common (and less complicated) tasks.
The Python API tries to be as easy to use as possible, but the CLI provides
wrappers around some parts of the Python API to make it easier to integrate
into non-Python workflows.

The ``openfe`` command consists of several subcommands. This is similar to
tools like ``git``, which has subcommands like ``git clone``, or ``conda``,
which has subcommands like ``conda install``. To get a list of the
subcommands and a brief description of them, use ``openfe --help`` (or
``openfe -h``), which will give:

.. code:: none

    Usage: openfe [OPTIONS] COMMAND [ARGS]...

      This is the command line tool to provide easy access to functionality from
      the OpenFE Python library.

    Options:
      --version   Show the version and exit.
      --log PATH  logging configuration file
      -h, --help  Show this message and exit.

    Setup Commands:
      plan-rhfe-network  Plan a relative hydration free energy network, saved in a
                         dir with multiple JSON files
      atommapping        Check the atom mapping of a given pair of ligands
      plan-rbfe-network  Plan a relative binding free energy network, saved in a
                         dir with multiple JSON files.

    Simulation Commands:
      gather    Gather DAG result jsons for network of RFE results into single TSV
                file
      quickrun  Run a given transformation, saved as a JSON file

The ``--log`` option takes a logging configuration file and sets that
logging behavior. If you use it, it must come before the subcommand name.

You can find out more about each subcommand by putting ``--help`` *after*
the subcommand name, e.g., ``openfe quickrun --help``, which returns

.. code:: none

    Usage: openfe quickrun [OPTIONS] TRANSFORMATION

      Run the transformation (edge) in the given JSON file in serial.

      To save a transformation as JSON, create the transformation and then save it
      with transformation.dump(filename).

    Options:
      -d, --work-dir DIRECTORY  directory to store files in (defaults to current
                                directory)
      -o FILE                   output file (JSON format) for the final results
      -h, --help                Show this message and exit.


For more details on various commands, see the :ref:`cli-reference`.
