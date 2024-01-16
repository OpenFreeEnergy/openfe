Installation
============

The page has information for installing ``openfe``, installing software
packages that integrate with ``openfe``, and testing that your ``openfe``
installation is working.

``openfe`` currently only works on POSIX system (macOS and UNIX/Linux). It
is tested against Python 3.9, 3.10, and 3.11.

When you install ``openfe`` through any of the methods described below, you
will install both the core library and the command line interface (CLI). 

If you already have a Mamba installation, you can install ``openfe`` with:

.. parsed-literal::

  mamba create -c conda-forge -n openfe_env openfe=\ |version|
  mamba activate openfe_env

Note that you must run the latter line in each shell session where you want to use ``openfe``. OpenFE recommends the Mamba package manager for most users as it is orders of magnitude faster than the default Conda package manager. Mamba is a drop in replacement for Conda.

Installation with ``mambaforge`` (recommended)
----------------------------------------------

.. _MambaForge: https://github.com/conda-forge/miniforge#mambaforge

We recommend installing ``openfe`` with `MambaForge`_ because it provides easy
installation of other software that ``openfe`` needs, such as OpenMM and
AmberTools. We recommend ``mambaforge`` because it is faster than ``conda`` and
comes preconfigured to use ``conda-forge``.

To install and configure ``mambaforge``, you need to know your operating
system, your machine architecture (output of ``uname -m``), and your shell
(in most cases, can be determined from ``echo $SHELL``). Select
your operating system and architecture from the tool below, and run the
commands it suggests.

.. raw:: html

    <select id="mambaforge-os" onchange="javascript: setArchitectureOptions(this.options[this.selectedIndex].value)">
        <option value="Linux">Linux</option>
        <option value="MacOSX">macOS</option>
    </select>
    <select id="mambaforge-architecture" onchange="updateInstructions()">
    </select>
    <select id="mambaforge-shell" onchange="updateInstructions()">
        <option value="bash">bash</option>
        <option value="zsh">zsh</option>
        <option value="tcsh">tcsh</option>
        <option value="fish">fish</option>
        <option value="xonsh">xonsh</option>
    </select>
    <br />
    <pre><span id="mambaforge-curl-install"></span></pre>
    <script>
      function setArchitectureOptions(os) {
          let options = {
              "MacOSX": [
                  ["x86_64", ""],
                  ["arm64", " (Apple Silicon)"]
              ],
              "Linux": [
                  ["x86_64", " (amd64)"],
                  ["aarch64", " (arm64)"],
                  ["ppc64le", " (POWER8/9)"]
              ]
          };
          choices = options[os];
          let htmlString = ""
          for (const [val, extra] of choices) {
              htmlString += `<option value="${val}">${val}${extra}</option>`;
          }
          let arch = document.getElementById("mambaforge-architecture");
          arch.innerHTML = htmlString
          updateInstructions()
      }

      function updateInstructions() {
          let cmd = document.getElementById("mambaforge-curl-install");
          let osElem = document.getElementById("mambaforge-os");
          let archElem = document.getElementById("mambaforge-architecture");
          let shellElem = document.getElementById("mambaforge-shell");
          let os = osElem[osElem.selectedIndex].value;
          let arch = archElem[archElem.selectedIndex].value;
          let shell = shellElem[shellElem.selectedIndex].value;
          let filename = "Mambaforge-" + os + "-" + arch + ".sh"
          let cmdArr = [
              (
                  "curl -OL https://github.com/conda-forge/miniforge/"
                  + "releases/latest/download/" + filename
              ),
              "sh " + filename + " -b",
              "~/mambaforge/bin/mamba init " + shell,
              "rm -f " + filename,
          ]
          cmd.innerHTML = cmdArr.join("\n")
      }

      setArchitectureOptions("Linux");  // default
    </script>

You should then close your current session and open a fresh login to ensure
that everything is properly registered.

Next we will create an environment called ``openfe_env`` with the ``openfe`` package and all required dependencies:

.. parsed-literal::

  mamba create -n openfe_env openfe=\ |version|

Now we need to activate our new environment ::

  mamba activate openfe_env


.. warning::

   Installing on Macs with Apple Silicon requires a creating an x86_64
   environment, as one of our requirements is not yet available for Apple
   Silicon. Run the following modified commands

   .. parsed-literal:: 

      CONDA_SUBDIR=osx-64 mamba create -n openfe_env openfe=\ |version|
      mamba activate openfe_env
      mamba env config vars set CONDA_SUBDIR=osx-64

To make sure everything is working, run the tests ::

  openfe test --long

The test suite contains several hundred individual tests. This may take up to
an hour, and all tests should complete with status either passed,
skipped, or xfailed (expected fail). The very first time you run this, the
initial check that you can import ``openfe`` will take a while, because some
code is compiled the first time it is encountered. That compilation only
happens once per installation.
  
With that, you should be ready to use ``openfe``!

Single file installer
---------------------

.. _releases on GitHub: https://github.com/OpenFreeEnergy/openfe/releases

Single file installers are available for x86_64 Linux and MacOS. 
They are attached to our `releases on GitHub`_ and can be downloaded with a browser or ``curl`` (or similar tool).
For example, the Linux installer can be downloaded with ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/latest/download/OpenFEforge-Linux-x86_64.sh

And the MacOS installer ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/latest/download/OpenFEforge-MacOSX-x86_64.sh

The single file installer contains all of the dependencies required for ``openfe`` and does not require internet access to use.

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

To make sure everything is working, run the tests ::

  $ pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).
  
With that, you should be ready to use ``openfe``!

Containers
----------

We provide an official docker and Apptainer (formerly Singularity) image.
The docker image is tagged with the version of ``openfe`` on the image and can be pulled with ::

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

You can access the ``openfe`` CLI from the Singularity image with ::

  $ singularity run --nv openfe_latest-apptainer.sif openfe --help

To make sure everything is working, run the tests ::

  $ singularity run --nv openfe_latest-apptainer.sif pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).
  
With that, you should be ready to use ``openfe``!

Developer install
-----------------

If you're going to be developing for ``openfe``, you will want an
installation where your changes to the code are immediately reflected in the
functionality. This is called a "developer" or "editable" installation.

Getting a developer installation for ``openfe`` first installing the
requirements, and then creating the editable installation. We recommend
doing that with ``mamba`` using the following procedure:

First, clone the ``openfe`` repository, and switch into its root directory::

  $ git clone https://github.com/OpenFreeEnergy/openfe.git
  $ cd openfe

Next create a ``conda`` environment containing the requirements from the
specification in that directory::

  $ mamba create -f environment.yml

Then activate the ``openfe`` environment with::

  $ mamba activate openfe_env

Finally, create the editable installation::

  $ python -m pip install --no-deps -e .

Note the ``.`` at the end of that command, which indicates the current
directory.

Optional dependencies
---------------------

Certain functionalities are only available if you also install other,
optional packages.

* **perses tools**: To use perses, you need to install perses and OpenEye,
  and you need a valid OpenEye license. To install both packages, use::

    $ mamba install -c openeye perses openeye-toolkits

HPC Environments
----------------

When using High Performance Computing resources, jobs are typically submitted to a queue from a "login node" and then run at a later time, often on different hardware and in a different software environment.
This can complicate installation as getting something working on the login node does not guarantee it will work in the job.
We recommend using `Apptainer (formerly Singularity) <https://apptainer.org/>`_ when running ``openfe`` workflows in HPC environments.
This images provide a software environment that is isolated from the host which can make workflow execution easier to setup and more reproducible.
See our guide on :ref:`containers <installation:containers>` for how to get started using Apptainer/Singularity.

.. _installation:mamba_hpc:

``mamba`` in HPC Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _virtual packages: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#managing-virtual-packages

We recommend using a :ref:`container <installation:containers>` to install ``openfe`` in HPC environments.
Nonetheless, ``openfe`` can be installed via Conda Forge on these environments also.
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
For example, to install a version of ``openfe`` which is compatible with ``cudatoolkit 11.7``, run:

.. parsed-literal::

  $ CONDA_OVERRIDE_CUDA=11.7 mamba create -n openfe_env openfe=\ |version|

Common Errors
-------------

openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)
  This error likely means that the CUDA version that ``openmm`` was built with is incompatible with the CUDA driver.
  Try re-making the environment while specifying the correct CUDA toolkit version for your hardware and driver.
  See :ref:`installation:mamba_hpc` for more details.
