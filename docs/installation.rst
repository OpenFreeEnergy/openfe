Installation
============

**openfe** is currently only compatible with POSIX systems (macOS and UNIX/Linux).

We try to follow `SPEC0 <https://scientific-python.org/specs/spec-0000/>`_ as far as minimum supported dependencies, with the following caveats:

- OpenMM 8.0, 8.1.2, 8.2.0 - **we do not yet support OpenMM v8.3.0**
- ``OpenEye Toolkits`` is not yet compatible with Python 3.13, so **openfe** cannot use openeye functionality with Python 3.13.

Note that following SPEC0 means that Python 3.10 support is no longer actively maintained as of ``openfe 1.6.0``.
Additionally, if you want to use NAGL to assign partial charges, you must use ``python >= 3.11``.

When you install **openfe** through any of the methods described below, you will install both the core library and the command line interface (CLI).

Installation with ``micromamba`` (recommended)
----------------------------------------------

OpenFE recommends ``mamba`` (and the more lightweight ``micromamba``) as a package manager for most users.
``mamba`` is drop-in replacement for ``conda`` and is orders of magnitude faster than the default ``conda`` package manager.
In the instructions below, we will use the ``micromamba`` command, but you can use ``conda`` or ``mamba`` in the same way.

Once you have one of `micromamba <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_, `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_, or `conda <https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html>`_ installed, you can continue to the **openfe** installation instructions below.


Reproducible builds with a ``conda-lock`` file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _conda-lock: https://github.com/conda/conda-lock?tab=readme-ov-file#conda-lock

We recommend building from **openfe**'s ``conda-lock`` file in most cases, since it allows for building packages in a reproducible way on multiple platforms.

Unlike the single file installer, an internet connection is required to install from a ``conda-lock`` file.

The ``conda-lock`` files for the latest version of **openfe** can be downloaded with ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/latest/download/openfe-conda-lock.yml

If a particular version is required, the URL will look like this (using the ``openfe 1.6.1`` release as an example) ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/download/v1.6.1/openfe-1.6.1-conda-lock.yml

``micromamba`` supports ``conda-lock`` files and can be used directly to create a virtual environment ::

    $ micromamba create -n openfe --file openfe-conda-lock.yml
    $ micromamba activate openfe


.. note::

   If you are having trouble building from the conda-lock file, you may need to build directly with ``conda-lock``.
   We recommend installing ``conda-lock`` in a new virtual environment.
   This will reduce the chance of dependency conflicts ::

       $ # Install conda lock into a virtual environment
       $ micromamba create -n conda-lock conda-lock
       $ # Activate the environment to use the conda-lock command
       $ micromamba activate conda-lock
       $ conda-lock install -n openfe openfe-conda-lock.yml
       $ micromamba activate openfe

To make sure everything is working, run the tests ::

  $ openfe test


The test suite contains several hundred individual tests.
This will take a few minutes, and all tests should complete with status either passed, skipped, or xfailed (expected fail).

Note that you must run ``micromamba activate openfe`` in each shell session where you want to use **openfe**. 

With that, you should be ready to use **openfe**!

Standard Installation with ``micromamba``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There may be some instances where you don't want to use a lock-file, e.g. you may want to specify a dependency that differs from the lock file.

In these cases, you can simply install **openfe** from conda-forge:

.. parsed-literal::

  micromamba create -c conda-forge -n openfe openfe=\ |version|
  micromamba activate openfe


Single file installer
---------------------

.. warning::

   The single file installer may modify your ``.bashrc`` in a way that requires manual intervention to access your previous ``conda`` installation

.. _releases on GitHub: https://github.com/OpenFreeEnergy/openfe/releases

Single file installers are available for x86_64 Linux and MacOS.
They are attached to our `releases on GitHub`_ and can be downloaded with a browser or ``curl`` (or similar tool).
For example, the Linux installer can be downloaded with ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/latest/download/OpenFEforge-Linux-x86_64.sh

And the MacOS (arm64) installer ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/latest/download/OpenFEforge-MacOSX-arm64.sh

MacOS x86_64 is no longer supported.

The single file installer contains all of the dependencies required for **openfe** and does not require internet access to use.

Both ``conda`` and ``mamba`` are also available in the environment created by the single file installer and can be used to install additional packages.
The installer can be installed in batch mode or interactively  ::

  $ chmod +x ./OpenFEforge-Linux-x86_64.sh # Make installer executable
  $ ./OpenFEforge-Linux-x86_64.sh # Run the installer

Example installer output is shown below (click to expand "Installer Output")

.. collapse:: Installer Output

  .. code-block::

      Welcome to OpenFEforge 0.7.4

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
     The install location will be different when you run the installer.

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

   Network Planning Commands:
     plan-rhfe-network    Plan a relative hydration free energy network, saved as
                          JSON files for the quickrun command.
     plan-rbfe-network    Plan a relative binding free energy network, saved as
                          JSON files for the quickrun command.
     view-ligand-network  Visualize a ligand network

   Quickrun Executor Commands:
     gather    Gather result jsons for network of RFE results into a TSV file
     quickrun  Run a given transformation, saved as a JSON file

   Miscellaneous Commands:
     fetch             Fetch tutorial or other resource.
     charge-molecules  Generate partial charges for a set of molecules.
     test              Run the OpenFE test suite




To make sure everything is working, run the tests ::

  $ pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).

With that, you should be ready to use **openfe**!

.. _installation:containers:

Containerized  Distributions
----------------------------

We provide an official docker and Apptainer (formerly Singularity) image.
The docker image is tagged with the version of **openfe** on the image and can be pulled with ::

  $ docker pull ghcr.io/openfreeenergy/openfe:latest

The Apptainer image is pre-built and can be pulled with ::

  $ singularity pull oras://ghcr.io/openfreeenergy/openfe:latest-apptainer

.. warning::

   For production use, we recommend using version tags to prevent disruptions in workflows e.g.

   .. parsed-literal::

     $ docker pull ghcr.io/openfreeenergy/openfe:\ |version|
     $ singularity pull oras://ghcr.io/openfreeenergy/openfe:\ |version|-apptainer

We recommend testing the container to ensure that it can access a GPU (if desired).
This can be done with the following command ::

  $ singularity run --nv openfe_latest-apptainer.sif python -m openmm.testInstallation

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

The ``--nv`` flag is required for the Apptainer image to access the GPU on the host.
Your output may produce different values for the forces, but should list the CUDA platform if everything is working properly.

You can access the **openfe** CLI from the Singularity image with ::

  $ singularity run --nv openfe_latest-apptainer.sif openfe --help

To make sure everything is working, run the tests ::

  $ singularity run --nv openfe_latest-apptainer.sif pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).

With that, you should be ready to use **openfe**!

.. note::

   If building a custom docker image, you may need to need to add ``--ulimit nofile=262144:262144`` to the ``docker build`` command.
   See this `issue <https://github.com/OpenFreeEnergy/openfe/issues/1269>`_ for details. 

HPC Environments
----------------

When using High Performance Computing resources, jobs are typically submitted to a queue from a "login node" and then run at a later time, often on different hardware and in a different software environment.
This can complicate installation as getting something working on the login node does not guarantee it will work in the job.
We recommend using `Apptainer (formerly Singularity) <https://apptainer.org/>`_ when running **openfe** workflows in HPC environments.
This images provide a software environment that is isolated from the host which can make workflow execution easier to setup and more reproducible.
See our guide on :ref:`containers <installation:containers>` for how to get started using Apptainer/Singularity.

.. _installation:mamba_hpc:

``mamba`` in HPC Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _virtual packages: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#managing-virtual-packages

We recommend using a :ref:`container <installation:containers>` to install **openfe** in HPC environments.
Nonetheless, **openfe** can be installed via Conda Forge on these environments also.
Conda Forge distributes its own CUDA binaries for interfacing with the GPU, rather than use the host drivers.
``conda``, ``mamba`` and ``micromamba`` all use `virtual packages`_ to detect and specify which version of CUDA should be installed.
This is a common point of difference in hardware between the login and job nodes in an HPC environment.
For example, on a login node where there likely is not a GPU or a CUDA environment, ``mamba info`` may produce output that looks like this ::

  $ mamba info

              mamba version : 1.5.1
         active environment : base
        active env location : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0
                shell level : 1
           user config file : /home/henrym3/.condarc
     populated config files : /lila/home/henrym3/.condarc
              conda version : 23.7.4
        conda-build version : not installed
             python version : 3.11.5.final.0
           virtual packages : __archspec=1=x86_64
                              __glibc=2.17=0
                              __linux=3.10.0=0
                              __unix=0=0
           base environment : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0  (writable)
          conda av data dir : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0/etc/conda
      conda av metadata url : None
               channel URLs : https://conda.anaconda.org/conda-forge/linux-64
                              https://conda.anaconda.org/conda-forge/noarch
              package cache : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0/pkgs
                              /home/henrym3/.conda/pkgs
           envs directories : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0/envs
                              /home/henrym3/.conda/envs
                   platform : linux-64
                 user-agent : conda/23.7.4 requests/2.31.0 CPython/3.11.5 Linux/3.10.0-957.12.2.el7.x86_64 centos/7.6.1810 glibc/2.17
                    UID:GID : 1987:3008
                 netrc file : None
               offline mode : False

Now if we run the same command on a HPC node that has a GPU ::

  $ mamba info

                mamba version : 1.5.1
         active environment : base
        active env location : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0
                shell level : 1
           user config file : /home/henrym3/.condarc
     populated config files : /lila/home/henrym3/.condarc
              conda version : 23.7.4
        conda-build version : not installed
             python version : 3.11.5.final.0
           virtual packages : __archspec=1=x86_64
                              __cuda=11.7=0
                              __glibc=2.17=0
                              __linux=3.10.0=0
                              __unix=0=0
           base environment : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0  (writable)
          conda av data dir : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0/etc/conda
      conda av metadata url : None
               channel URLs : https://conda.anaconda.org/conda-forge/linux-64
                              https://conda.anaconda.org/conda-forge/noarch
              package cache : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0/pkgs
                              /home/henrym3/.conda/pkgs
           envs directories : /lila/home/henrym3/mamba/envs/QA-openfe-0.14.0/envs
                              /home/henrym3/.conda/envs
                   platform : linux-64
                 user-agent : conda/23.7.4 requests/2.31.0 CPython/3.11.5 Linux/3.10.0-1160.45.1.el7.x86_64 centos/7.9.2009 glibc/2.17
                    UID:GID : 1987:3008
                 netrc file : None
               offline mode : False


We can see that there is a virtual package ``__cuda=11.7=0``.
This means that if we run a ``mamba install`` command on a node with a GPU, the solver will install the correct version of the ``cudatoolkit``.
However, if we ran the same command on the login node, the solver may install the wrong version of the ``cudatoolkit``, or depending on how the Conda packages are setup, a CPU only version of the package.
We can control the virtual package with the environmental variable ``CONDA_OVERRIDE_CUDA``.

In order to determine the correct ``cudatoolkit`` version, we recommend connecting to the node where the simulation will be executed and run ``nvidia-smi``.
For example ::

  $ nvidia-smi
  Tue Jun 13 17:47:11 2023
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 515.43.04    Driver Version: 515.43.04    CUDA Version: 11.7     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  NVIDIA A40          On   | 00000000:65:00.0 Off |                    0 |
  |  0%   30C    P8    32W / 300W |      0MiB / 46068MiB |      0%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+

  +-----------------------------------------------------------------------------+
  | Processes:                                                                  |
  |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
  |        ID   ID                                                   Usage      |
  |=============================================================================|
  |  No running processes found                                                 |
  +-----------------------------------------------------------------------------+

in this output of ``nvidia-smi`` we can see in the upper right of the output ``CUDA Version: 11.7`` which means the installed driver will support a ``cudatoolkit`` version up to ``11.7``

So on the login node, we can run ``CONDA_OVERRIDE_CUDA=11.7 mamba info`` and see that the "correct" virtual CUDA is listed.
For example, to install a version of **openfe** which is compatible with ``cudatoolkit 11.7``, run:

.. parsed-literal::

  $ CONDA_OVERRIDE_CUDA=11.7 mamba create -n openfe openfe=\ |version|

Developer install
-----------------

If you're going to be developing for **openfe**, you will want an
installation where your changes to the code are immediately reflected in the
functionality. This is called a "developer" or "editable" installation.

Getting a developer installation for **openfe** first installing the
requirements, and then creating the editable installation. We recommend
doing that with ``mamba`` using the following procedure:

First, clone the **openfe** repository, and switch into its root directory::

  $ git clone https://github.com/OpenFreeEnergy/openfe.git
  $ cd openfe

Next create a ``conda`` environment containing the requirements from the
specification in that directory::

  $ mamba create -f environment.yml

Then activate the **openfe** environment with::

  $ mamba activate openfe

Finally, create the editable installation::

  $ python -m pip install --no-deps -e .

Note the ``.`` at the end of that command, which indicates the current
directory.

Troubleshooting Your Installation
---------------------------------

We have created a script that can be run locally to assist in troubleshooting errors.
The script does not upload any information and the output may be inspected before the output is sent to us.
We recommend running the script in the same environment where the error was observed.
For example, if you had an error when creating a system on your local workstation, run the script locally with the same conda environment active as when the error occurred.
If the error occurred when running the job on an HPC resource, then run the script (ideally) on the same node where the problem occurred.
This helps to debug issues such as a CUDA and NVIDIA driver mismatch (which would be impossible to diagnose if the script was ran on a login node without a GPU).

The script is available here: https://github.com/OpenFreeEnergy/openfe/blob/main/devtools/debug_openmm.sh
For your convenience, this command will download the script and save the output as ``debug.log``

.. parsed-literal::

  $ bash -c "$(curl -Ls https://raw.githubusercontent.com/OpenFreeEnergy/openfe/main/devtools/debug_openmm.sh)" | tee -a debug.log

The output of the script will also be printed to standard out as it is executed.
While no sensitive information is extracted, it is good practice to review the output before sending it or posting it to ensure that nothing needs to be redacted.
For example, if your python path was ``/data/SECRET_COMPOUND_NAME/python`` then that would show up in ``debug.log``.


Common Errors
-------------

.. parsed-literal::

  openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)

This error likely means that the CUDA version that ``openmm`` was built with is incompatible with the CUDA driver.
Try re-making the environment while specifying the correct CUDA toolkit version for your hardware and driver.
See :ref:`installation:mamba_hpc` for more details.

Optional dependencies
---------------------

Certain functionalities are only available if you also install other,
optional packages.

* **perses tools**: To use perses, you need to install perses and OpenEye,
  and you need a valid OpenEye license. To install both packages, use::

    $ mamba install -c openeye perses openeye-toolkits

Supported Hardware
------------------

We currently support the following CPU architectures:

* ``linux-64``
* ``osx-arm64``

For simulation preparation, any supported platform is suitable.
We test our software regularly by performing vacuum transformations on ``linux-64`` using the OpenMM CUDA platform.
While OpenMM supports OpenCL, we do not regularly test that platform (the CUDA platform is more performant) so we do not recommend using that platform without performing your own verification of correctness.
For production use, we recommend the ``linux-64`` platform with NVIDIA GPUs for optimal performance.
When using an OpenMM based protocol on NVIDIA GPUs, we recommend driver version ``525.60.13`` or greater.
The minimum driver version required when installing from conda-forge is ``450.36.06``, but newer versions of OpenMM may not support that driver version as CUDA 11 will be removed the build matrix.
