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

Single file installers are available for x86_64 Linux and MacOS. 
They are attached to our `releases on GitHub <https://github.com/OpenFreeEnergy/openfe/releases>`_ and can be downloaded with a browser or ``curl`` (or similar tool).
For example, the linux installer can be downloaded with ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/download/v0.7.1/OpenFEforge-0.7.1-Linux-x86_64.sh

And the MacOS installer ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/download/v0.7.1/OpenFEforge-0.7.1-MacOSX-x86_64.sh 

The single file installer contains all of the dependencies required for ``openfe`` and does not require internet access to use.
Both ``conda`` and ``mamba`` are also available in the environment created by the single file installer and can be used to install additional packages.
The installer can be installed in batch mode or interactively  ::
  
  $ chmod +x ./OpenFEforge-0.7.1-Linux-x86_64.sh # Make installer executable
  $ ./OpenFEforge-0.7.1-Linux-x86_64.sh # Run the installer

Example installer output is shown below (click to expand "Installer Output")

.. collapse:: Installer Output

  .. code-block::
  
      Welcome to OpenFEforge 0.7.1
    
      In order to continue the installation process, please review the license
      agreement.
      Please, press ENTER to continue
      >>>
      MIT License
    
      Copyright (c) 2022 OpenFreeEnergy
      
      Permission is hereby granted, free of charge, to any person obtaining a copy
      of this software and associated documentation files (the "Software"), to deal
      in the Software without restriction, including without limitation the rights
      to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
      copies of the Software, and to permit persons to whom the Software is
      furnished to do so, subject to the following conditions:
      
      The above copyright notice and this permission notice shall be included in all
      copies or substantial portions of the Software.
      
      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
      IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
      FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
      LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
      OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
      SOFTWARE.
      
      
      Do you accept the license terms? [yes|no]
      [no] >>> yes
  
  .. note:: 
     Your path will be different 
     
  
  .. code-block::
    
      OpenFEforge will now be installed into this location:
      /home/mmh/openfeforge
    
      - Press ENTER to confirm the location
      - Press CTRL-C to abort the installation
      - Or specify a different location below
    
      [/home/mmh/openfeforge] >>>
      PREFIX=/home/mmh/openfeforge
      Unpacking payload ...
      
      Installing base environment...
      
      
      Downloading and Extracting Packages
      
      
      Downloading and Extracting Packages
      
      Preparing transaction: done
      Executing transaction: \ By downloading and using the CUDA Toolkit conda packages, you accept the terms and conditions of the CUDA End User License Agreement (EULA): https://docs.nvidia.com/cuda/eula/index.html
      
      | Enabling notebook extension jupyter-js-widgets/extension...
            - Validating: OK
      
      done
      installation finished.
      Do you wish the installer to initialize OpenFEforge
      by running conda init? [yes|no]
      [no] >>> yes
      no change     /home/mmh/openfeforge/condabin/conda
      no change     /home/mmh/openfeforge/bin/conda
      no change     /home/mmh/openfeforge/bin/conda-env
      no change     /home/mmh/openfeforge/bin/activate
      no change     /home/mmh/openfeforge/bin/deactivate
      no change     /home/mmh/openfeforge/etc/profile.d/conda.sh
      no change     /home/mmh/openfeforge/etc/fish/conf.d/conda.fish
      no change     /home/mmh/openfeforge/shell/condabin/Conda.psm1
      no change     /home/mmh/openfeforge/shell/condabin/conda-hook.ps1
      no change     /home/mmh/openfeforge/lib/python3.9/site-packages/xontrib/conda.xsh
      no change     /home/mmh/openfeforge/etc/profile.d/conda.csh
      modified      /home/mmh/.bashrc
      
      ==> For changes to take effect, close and re-open your current shell. <==
      
      
                        __    __    __    __
                       /  \  /  \  /  \  /  \
                      /    \/    \/    \/    \
      ███████████████/  /██/  /██/  /██/  /████████████████████████
                    /  / \   / \   / \   / \  \____
                   /  /   \_/   \_/   \_/   \    o \__,
                  / _/                       \_____/  `
                  |/
              ███╗   ███╗ █████╗ ███╗   ███╗██████╗  █████╗
              ████╗ ████║██╔══██╗████╗ ████║██╔══██╗██╔══██╗
              ██╔████╔██║███████║██╔████╔██║██████╔╝███████║
              ██║╚██╔╝██║██╔══██║██║╚██╔╝██║██╔══██╗██╔══██║
              ██║ ╚═╝ ██║██║  ██║██║ ╚═╝ ██║██████╔╝██║  ██║
              ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝
      
              mamba (1.4.2) supported by @QuantStack
      
              GitHub:  https://github.com/mamba-org/mamba
              Twitter: https://twitter.com/QuantStack
      
      █████████████████████████████████████████████████████████████
      
      no change     /home/mmh/openfeforge/condabin/conda
      no change     /home/mmh/openfeforge/bin/conda
      no change     /home/mmh/openfeforge/bin/conda-env
      no change     /home/mmh/openfeforge/bin/activate
      no change     /home/mmh/openfeforge/bin/deactivate
      no change     /home/mmh/openfeforge/etc/profile.d/conda.sh
      no change     /home/mmh/openfeforge/etc/fish/conf.d/conda.fish
      no change     /home/mmh/openfeforge/shell/condabin/Conda.psm1
      no change     /home/mmh/openfeforge/shell/condabin/conda-hook.ps1
      no change     /home/mmh/openfeforge/lib/python3.9/site-packages/xontrib/conda.xsh
      no change     /home/mmh/openfeforge/etc/profile.d/conda.csh
      no change     /home/mmh/.bashrc
      No action taken.
      Added mamba to /home/mmh/.bashrc
      
      ==> For changes to take effect, close and re-open your current shell. <==
      
      If you'd prefer that conda's base environment not be activated on startup,
         set the auto_activate_base parameter to false:
      
      conda config --set auto_activate_base false
      
      Thank you for installing OpenFEforge!
  
After the installer completes, close and reopen your shell. 
To check if your path is setup correctly, run ``which python`` your output should look something like this ::

   (base) $ which python
   /home/mmh/openfeforge/bin/python

.. note::
   Your path will be different, but the important part is ``openfeforge/bin/python``

Now the CLI tool should work as well ::

   (base) $ openfe --help
   Usage: openfe [OPTIONS] COMMAND [ARGS]...
   
     This is the command line tool to provide easy access to functionality from
     the OpenFE Python library.
   
   Options:
     --version   Show the version and exit.
     --log PATH  logging configuration file
     -h, --help  Show this message and exit.
   
   Setup Commands:
     atommapping        Check the atom mapping of a given pair of ligands
     plan-rhfe-network  Plan a relative hydration free energy network, saved in a
                        dir with multiple JSON files
     plan-rbfe-network  Plan a relative binding free energy network, saved in a
                        dir with multiple JSON files.
   
   Simulation Commands:
     gather    Gather DAG result jsons for network of RFE results into single TSV
               file
     quickrun  Run a given transformation, saved as a JSON file

 

Containers
~~~~~~~~~~

We provide an official docker and apptainer (formally singularity) image.
The docker image is tagged with the version of ``openfe`` on the image and can be pulled with ::

  $ docker pull ghcr.io/openfreeenergy/openfe:0.7.1

The apptainer image is pre-built and attached to our `releases on GitHub <https://github.com/OpenFreeEnergy/openfe/releases>`_ and can be downloaded with ``curl`` (or similar tool) ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/download/v0.7.1/openfe_0.7.1.sif

We recommend testing the container to ensure that it can access a GPU (if desired).
This can be done with the following command ::

  $ singularity run --nv openfe_0.7.1.sif python -m openmm.testInstallation
  
  OpenMM Version: 8.0
  Git Revision: a7800059645f4471f4b91c21e742fe5aa4513cda

  There are 3 Platforms available:

  1 Reference - Successfully computed forces
  2 CPU - Successfully computed forces
  3 CUDA - Successfully computed forces

  Median difference in forces between platforms:

  Reference vs. CPU: 6.29328e-06
  Reference vs. CUDA: 6.7337e-06
  CPU vs. CUDA: 7.44698e-07

  All differences are within tolerance.

The ``--nv`` flag is required for the apptainer image to access the GPU on the host. 
Your output may produce different values for the forces, but should list the CUDA platform if everything is working properly. 

You can access the ``openfe`` CLI from the singularity image with ::

  $ singularity run --nv openfe_0.7.1.sif openfe --help

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
