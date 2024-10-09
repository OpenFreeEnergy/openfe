Tutorials
=========

.. todo: make sure we can inline the tutorial, for now we only provide links

OpenFE has several tutorial notebooks which are maintained on our
`Example Notebooks repository <https://github.com/OpenFreeEnergy/ExampleNotebooks>`_.

Here is a list of key tutorials which cover the different aspects of the
OpenFE tooling:


Relative Free Energies
----------------------

Python API Showcase
~~~~~~~~~~~~~~~~~~~

Our :any:`showcase notebook <showcase_notebook>` walks users through
how to use the main Python API components of OpenFE to create a
relative binding free energy calculation.

Relative Free Energies CLI tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :ref:`Relative Free Energies with the OpenFE CLI <rbfe_cli_tutorial>`
tutorial walks users through how to use the OpenFE command line to calculate
relative binding free energies of various ligands against the TYK2 target.

Associated with it is also a :any:`notebook <rbfe_python_tutorial>`
for how to achieve the same outcomes using the Python API.


Absolute Free Energies
----------------------

Absolute Solvation Free Energies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :any:`Absolute Solvation Free Energy Protocol <ahfe_tutorial>` tutorial
walks users through how to calculate the hydration free energy of a benzene
ligand.


Molecular Dynamics (MD)
-----------------------

The :any:`MD protocol <md_tutorial>`
tutorial walks users through how to run a conventional (non-alchemical) MD
simulation of benzene bound to T4-lysozyme L99A in OpenFE.


Post Simulation Analysis
------------------------

Analyzing and Plotting RFE Networks with Cinnabar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :any:`Cinnabar tutorial <plotting_with_cinnabar>` demonstrates how to
use the `Cinnabar Python package <https://github.com/OpenFreeEnergy/cinnabar>`_
to analyze (e.g. generating MLE estimates of absolute free energies)
and plot networks of relative free energy results.


.. toctree::
    :maxdepth: 1
    :hidden:
    
    rbfe_cli_tutorial
    rbfe_python_tutorial
    showcase_notebook
    md_tutorial
    ahfe_tutorial
    plotting_with_cinnabar
