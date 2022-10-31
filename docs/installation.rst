Installation
============

The page has information for installing ``openfe``, installing software
packages that integrate with ``openfe``, and testing that your ``openfe``
installation is working.

``openfe`` currently only works on POSIX system (macOS and UNIX/Linux). It
is tested against Python 3.9 and 3.10.

Installing ``openfe``
---------------------

When you install ``openfe`` through any of the methods described below, you
will install both the core library and the command line interface (CLI). 

Installation with ``conda`` (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing ``openfe`` with ``conda``, because it provides easy
installation of other tools, including molecular dynamics tools such as
OpenMM and ambertools, which are needed by ``openfe``. 
Conda can be installed either with the `full Anaconda distribution
<https://www.anaconda.com/products/individual>`_, or with
the `smaller-footprint miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_.

You can get ``openfe`` from the ``conda-forge`` channel. To install the most
recent release in its own environment, use the commands ::

  $ conda create -n openfe -c conda-forge openfe
  $ conda activate openfe

That creates a separate environment for ``openfe`` that won't interfere with
other things you have installed. You will need to activate the ``openfe``
environment before using it in a new shell.

With that, you should be ready to use ``openfe``!

Single file installer
~~~~~~~~~~~~~~~~~~~~~

[COMING SOON!]

.. TODO: maybe Mike can fill this in? just needs (1) how to download the
   single file installer; (2) how to use the single file installer

Developer install
~~~~~~~~~~~~~~~~~

If you're going to be developing for ``openfe``, you will want an
installation where your changes to the code are immediately reflected in the
functionality. This is called a "developer" or "editable" installation.

Getting a developer installation for ``openfe`` first installing the
requirements, and then creating the editable installation. We recommend
doing that with ``conda`` using the following procedure:

First, clone the ``openfe`` repository, and switch into its root directory::

  $ git clone https://github.com/OpenFreeEnergy/openfe.git
  $ cd openfe

Next create a ``conda`` environment containing the requirements from the
specification in that directory::

  $ conda env create -f environment.yml

Then activate the ``openfe`` environment with::

  $ conda activate openfe

Finally, create the editable installation::

  $ python -m pip install -e .

Note the ``.`` at the end of that command, which indicates the current
directory.

Optional dependencies
---------------------

Certain functionalities are only available if you also install other,
optional packages.

* **perses tools**: To use perses, you need to install perses and OpenEye,
  and you need a valid OpenEye license. To install both packages, use::

    $ conda install -c conda-forge -c openeye perses openeye-toolkits

Testing your installation
-------------------------

``openfe`` has a thorough test suite, and running the test suite is a good
start to troubleshooting any installation problems. The test suite requires
``pytest`` to run. You can install ``pytest`` with::

  $ conda install -c conda-forge  pytest

Then you can run the test suite (from any directory) with the command::

  $ pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).
