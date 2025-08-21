Tutorials
=========

.. todo: make sure we can inline the tutorial, for now we only provide links

Below is a collection of tutorials that demonstrate key elements of OpenFE tooling.

You can clone the `Example Notebooks Repository <https://github.com/OpenFreeEnergy/ExampleNotebooks>`_ to explore any of these tutorials interactively.


Relative Free Energies
----------------------

- :any:`Python API Showcase <showcase_notebook>`: Start here! An introduction to OpenFE's Python API and approach to performing a relative binding free energy calculation.
- :any:`RBFE with the Python API <rbfe_python_tutorial>`: A step-by-step tutorial for using the Python API to calculate relative binding free energies for TYK2.
- :ref:`RBFE with the CLI <rbfe_cli_tutorial>`: A step-by-step tutorial for using the OpenFE command line interface (CLI) to calculate relative binding free energies for TYK2.

Absolute Free Energies
----------------------

- :any:`Absolute Solvation Free Energy Protocol <ahfe_tutorial>`: A walk-through of calculating the hydration free energy of a benzene ligand.

Relative Free Energies using Separated Topologies
-------------------------------------------------

- :any:`SepTop Protocol <septop_tutorial>`: A walk-through of calculating the relative binding free energy between TYK2 ligands using a Separated Topologies approach.

Molecular Dynamics (MD)
-----------------------

- :any:`MD protocol <md_tutorial>`: A walk-through of running a conventional (non-alchemical) MD simulation of benzene bound to T4-lysozyme L99A.

Post-Simulation Analysis
------------------------

- :any:`Cinnabar tutorial <plotting_with_cinnabar>`: A tutorial for using the `cinnabar <https://github.com/OpenFreeEnergy/cinnabar>`_ Python package to analyze (e.g. generating MLE estimates of absolute free energies) and plot networks of relative free energy results.

Generating Partial Charges
--------------------------

.. todo: this should be in cookbook

-  :ref:`Generating Partial Charges CLI tutorial <charge_molecules_cli_tutorial>`: how to use the CLI to assign and store partial charges for mall molecules which can be used throughout the OpenFE ecosystem.

.. toctree::
    :maxdepth: 1
    :hidden:

    showcase_notebook
    rbfe_python_tutorial
    rbfe_cli_tutorial
    ahfe_tutorial
    septop_tutorial
    md_tutorial
    plotting_with_cinnabar
    charge_molecules_cli_tutorial
