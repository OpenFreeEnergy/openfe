#!/bin/bash

#SBATCH --job-name="openfe job"
#SBATCH --mem-per-cpu=2G

# activate an appropriate conda environment, or any "module load" commands required to
conda activate openfe_env

# continue submitting run-task in serial until the wall time is hit
# you may submit this *script* multiple times to have workers execute tasks in parallel
while true; do
    openfe run-task --warehouse my_campaign/ --task-db my_campaign.db --scratch workdir/
done
