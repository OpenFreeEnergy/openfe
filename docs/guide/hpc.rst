HPC
===

We recommend using `apptainer (formally singularity) <https://apptainer.org/>`_ when running ``openfe`` workflows in HPC environments.
This images provide a software environment that is isolated from the host which can make workflow execution easier to setup and more reproducible.
See our guide on :ref:`containers <installation:containers>` for how to get started using apptainer/singularity.

``micromamba`` Installation Considerations in HPC Environments 
--------------------------------------------------------------

``conda``, ``mamba`` and ``micromamba`` all use `virtual packages <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#managing-virtual-packages>`_ to detect which version of CUDA should be installed.
For example, on a login node where there likely is not a GPU or a CUDA environment, ``micromamba info`` may produce output that looks like this ::

  $ micromamba info
 
                                            __
           __  ______ ___  ____ _____ ___  / /_  ____ _
          / / / / __ `__ \/ __ `/ __ `__ \/ __ \/ __ `/
         / /_/ / / / / / / /_/ / / / / / / /_/ / /_/ /
        / .___/_/ /_/ /_/\__,_/_/ /_/ /_/_.___/\__,_/
       /_/
 
 
             environment : openfe_env (active)
            env location : /lila/home/henrym3/micromamba/envs/openfe_env
       user config files : /home/henrym3/.mambarc
  populated config files : /home/henrym3/.condarc
        libmamba version : 1.2.0
      micromamba version : 1.2.0
            curl version : libcurl/7.87.0 OpenSSL/1.1.1s zlib/1.2.13 libssh2/1.10.0 nghttp2/1.47.0
      libarchive version : libarchive 3.6.2 zlib/1.2.13 bz2lib/1.0.8 libzstd/1.5.2
        virtual packages : __unix=0=0
                           __linux=3.10.0=0
                           __glibc=2.17=0
                           __archspec=1=x86_64
                channels : https://conda.anaconda.org/conda-forge/linux-64
                           https://conda.anaconda.org/conda-forge/noarch
        base environment : /lila/home/henrym3/micromamba
                platform : linux-64
 

Now if we run the same command on a HPC node that has a GPU ::

  $ micromamba info
 
                                            __
           __  ______ ___  ____ _____ ___  / /_  ____ _
          / / / / __ `__ \/ __ `/ __ `__ \/ __ \/ __ `/
         / /_/ / / / / / / /_/ / / / / / / /_/ / /_/ /
        / .___/_/ /_/ /_/\__,_/_/ /_/ /_/_.___/\__,_/
       /_/
 
 
             environment : openfe_env (active)
            env location : /lila/home/henrym3/micromamba/envs/openfe_env
       user config files : /home/henrym3/.mambarc
  populated config files : /home/henrym3/.condarc
        libmamba version : 1.2.0
      micromamba version : 1.2.0
            curl version : libcurl/7.87.0 OpenSSL/1.1.1s zlib/1.2.13 libssh2/1.10.0 nghttp2/1.47.0
      libarchive version : libarchive 3.6.2 zlib/1.2.13 bz2lib/1.0.8 libzstd/1.5.2
        virtual packages : __unix=0=0
                           __linux=3.10.0=0
                           __glibc=2.17=0
                           __archspec=1=x86_64
                           __cuda=11.7=0
                channels : https://conda.anaconda.org/conda-forge/linux-64
                           https://conda.anaconda.org/conda-forge/noarch
        base environment : /lila/home/henrym3/micromamba
                platform : linux-64

We can see that there is a virtual package ``__cuda=11.7=0``.
This means that if we run a ``micromamba install`` command on a node with a GPU, the solver will install the correct version of the ``cudatoolkit``.
However, if we ran the same command on the login node, the solver may install the wrong version of the ``cudatoolkit``, or depending on how the conda packages are setup, a CPU only version of the package.
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

So on the login node, we can run ``CONDA_OVERRIDE_CUDA=11.7 micromamba info`` and see that the "correct" virtual CUDA is listed.
For example, to install a version of ``openfe`` which is compatible with ``cudatoolkit 11.7`` run ``CONDA_OVERRIDE_CUDA=11.7 micromamba install openfe``.

Common Errors
-------------

openmm.OpenMMException: Error loading CUDA module: CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This error likely means that the CUDA version that ``openmm`` was built with is incompatible with the CUDA driver.
Try re-making the environment while specifying the CUDA toolkit version that works with the CUDA driver on the node.
For example ``micromamba create -n openfe_env openfe cudatoolkit==11.3``. 
