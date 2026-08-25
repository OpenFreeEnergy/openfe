#!/bin/bash

#SBATCH --job-name="openfe job"
#SBATCH --mem-per-cpu=2G

# activate an appropriate conda environment, or any "module load" commands required to
conda activate openfe_env

# continue submitting workers until the wall time is hit
while true; do  
    openfe run worker --warehouse my_campaign/ --task-db my_campaign.db --scratch workdir/
done
