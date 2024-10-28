CLI basics
==========

The ``openfe`` command consists of several subcommands. This is similar to
tools like ``gmx``, which has subcommands like ``gmx mdrun``, or ``conda``,
which has subcommands like ``conda install``.

To get a list of the subcommands and their descriptions, call ``openfe`` (or
``openfe -h``):

.. TODO autogenerate using sphinxcontrib-programoutput

.. code:: none

    Usage: openfe [OPTIONS] COMMAND [ARGS]...
    
      This is the command line tool to provide easy access to functionality from
      the OpenFE Python library.
    
    Options:
      --version   Show the version and exit.
      --log PATH  logging configuration file
      -h, --help  Show this message and exit.
    
    Network Planning Commands:
      view-ligand-network  Visualize a ligand network
      plan-rbfe-network    Plan a relative binding free energy network, saved as
                           JSON files for the quickrun command.
      plan-rhfe-network    Plan a relative hydration free energy network, saved as
                           JSON files for the quickrun command.
    
    Quickrun Executor Commands:
      gather    Gather result jsons for network of RFE results into a TSV file
      quickrun  Run a given transformation, saved as a JSON file
    
    Miscellaneous Commands:
      fetch  Fetch tutorial or other resource.
      test   Run the OpenFE test suite

The ``--log`` option takes a logging configuration file and sets that
logging behavior. If you use it, it must come before the subcommand name.

You can find out more about each subcommand by putting ``--help`` *after*
the subcommand name, e.g., ``openfe quickrun --help``, which returns

.. code:: none

    Usage: openfe quickrun [OPTIONS] TRANSFORMATION
    
      Run the transformation (edge) in the given JSON file.
    
      Simulation JSON files can be created with the :ref:`cli_plan-rbfe-network`
      or from Python a :class:`.Transformation` can be saved using its dump
      method::
    
          transformation.dump("filename.json")
    
      That will save a JSON file suitable to be input for this command.
    
      Running this command will execute the simulation defined in the JSON file,
      creating a directory for each individual task (``Unit``) in the workflow.
      For example, when running the OpenMM HREX Protocol a directory will be
      created for each repeat of the sampling process (by default 3).
    
    Options:
      -d, --work-dir DIRECTORY  directory to store files in (defaults to current
                                directory)
      -o FILE                   output file (JSON format) for the final results
      -h, --help                Show this message and exit.

For more details on various commands, see the :ref:`cli-reference`.
