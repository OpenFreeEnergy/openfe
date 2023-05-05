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

Installation with ``mambaforge`` (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend installing ``openfe`` with `mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_, because it provides easy
installation of other tools, including molecular dynamics tools such as
OpenMM and ambertools, which are needed by ``openfe``.
We recommend ``mambaforge`` because it is faster than ``conda`` and comes
preconfigured to use ``conda-forge``.

To install ``mambaforge``, select your operating system and architecture from
the tool below, and run the ``curl`` / ``sh`` command it suggests.

.. someone else improve the above wording?

.. raw:: html

    <select id="mambaforge-os" onchange="javascript: setArchitectureOptions(this.options[this.selectedIndex].value)">
        <option value="Linux">Linux</option>
        <option value="MacOSX">macOS</option>
    </select>
    <select id="mambaforge-architecture" onchange="updateFilename()">
    </select>
    <br />
    <pre><span id="mambaforge-curl-install"></span></pre>
    <script>
      function setArchitectureOptions(os) {
          let options = {
              "MacOS": [
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
          updateFilename()
      }

      function updateFilename() {
          let cmd = document.getElementById("mambaforge-curl-install");
          let osElem = document.getElementById("mambaforge-os");
          let archElem = document.getElementById("mambaforge-architecture");
          let os = osElem[osElem.selectedIndex].value;
          let arch = archElem[archElem.selectedIndex].value;
          let filename = "Mambaforge-" + os + "-" + arch + ".sh"
          let cmdText = (
              "curl https://github.com/conda-forge/miniforge/releases/"
              + "latest/download/" + filename + " | sh"
          )
          cmd.innerHTML = cmdText

      }

      setArchitectureOptions("Linux");
    </script>

This command will create an environment called ``openfe_env`` with the ``openfe`` package and all required  dependencies ::

  $ mamba create -n openfe_env openfe

Now we need to activate our new environment ::

  $ mamba activate openfe_env

To make sure everything is working, run the tests ::

  $ openfe test --long

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).
  
With that, you should be ready to use ``openfe``!

Single file installer
^^^^^^^^^^^^^^^^^^^^^

Single file installers are available for x86_64 Linux and MacOS. 
They are attached to our `releases on GitHub <https://github.com/OpenFreeEnergy/openfe/releases>`_ and can be downloaded with a browser or ``curl`` (or similar tool).
For example, the linux installer can be downloaded with ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/download/v0.7.4/OpenFEforge-0.7.4-Linux-x86_64.sh

And the MacOS installer ::

  $ curl -LOJ https://github.com/OpenFreeEnergy/openfe/releases/download/v0.7.4/OpenFEforge-0.7.4-MacOSX-x86_64.sh 

The single file installer contains all of the dependencies required for ``openfe`` and does not require internet access to use.
Both ``conda`` and ``mamba`` are also available in the environment created by the single file installer and can be used to install additional packages.
The installer can be installed in batch mode or interactively  ::
  
  $ chmod +x ./OpenFEforge-0.7.4-Linux-x86_64.sh # Make installer executable
  $ ./OpenFEforge-0.7.4-Linux-x86_64.sh # Run the installer

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

To make sure everything is working, run the tests ::

  $ pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).
  
With that, you should be ready to use ``openfe``!

Containers
^^^^^^^^^^

We provide an official docker and apptainer (formally singularity) image.
The docker image is tagged with the version of ``openfe`` on the image and can be pulled with ::

  $ docker pull ghcr.io/openfreeenergy/openfe:0.7.4

The apptainer image is pre-built and can be pulled with ::

  $ singularity pull oras://ghcr.io/openfreeenergy/openfe:0.7.4-apptainer

We recommend testing the container to ensure that it can access a GPU (if desired).
This can be done with the following command ::

  $ singularity run --nv openfe_0.7.4-apptainer.sif python -m openmm.testInstallation
  
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

  $ singularity run --nv openfe_0.7.4-apptainer.sif openfe --help

To make sure everything is working, run the tests ::

  $ singularity run --nv openfe_0.7.4-apptainer.sif pytest --pyargs openfe openfecli

The test suite contains several hundred individual tests. This will take a
few minutes, and all tests should complete with status either passed,
skipped, or xfailed (expected fail).
  
With that, you should be ready to use ``openfe``!

Developer install
^^^^^^^^^^^^^^^^^

If you're going to be developing for ``openfe``, you will want an
installation where your changes to the code are immediately reflected in the
functionality. This is called a "developer" or "editable" installation.

Getting a developer installation for ``openfe`` first installing the
requirements, and then creating the editable installation. We recommend
doing that with ``micromamba`` using the following procedure:

First, clone the ``openfe`` repository, and switch into its root directory::

  $ git clone https://github.com/OpenFreeEnergy/openfe.git
  $ cd openfe

Next create a ``conda`` environment containing the requirements from the
specification in that directory::

  $ micromamba create -f environment.yml

Then activate the ``openfe`` environment with::

  $ micromamba activate openfe_env

Finally, create the editable installation::

  $ python -m pip install --no-deps -e .

Note the ``.`` at the end of that command, which indicates the current
directory.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

Certain functionalities are only available if you also install other,
optional packages.

* **perses tools**: To use perses, you need to install perses and OpenEye,
  and you need a valid OpenEye license. To install both packages, use::

    $ micromamba install -c openeye perses openeye-toolkits
