.. _howto_share_gpu_with_mps:

Share a GPU between runs with NVIDIA MPS
========================================

A single ``openfe quickrun`` call often does not use the whole GPU, particularly for small systems such as solvent legs.
NVIDIA's Multi-Process Service (MPS) lets several quickrun processes run on the same GPU at the same time.
Each run is slower than on its own, but the total throughput is higher.
See NVIDIA's documentation on
`MPS for OpenFE free energy calculations <https://developer.nvidia.com/blog/maximizing-openmm-molecular-dynamics-throughput-with-nvidia-multi-process-service/?ref=blog.omsf.io#mps_for_openfe_free_energy_calculations>`_ for details.

See also
--------

* :doc:`run_on_slurm` – submitting quickrun jobs to a cluster.
* :doc:`run_repeats_in_parallel` – running each repeat as a separate job.
