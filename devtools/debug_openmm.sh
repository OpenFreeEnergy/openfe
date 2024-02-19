#!/usr/bin/env bash

echo "Run this script with you conda env activated"
echo "Invoke the script like this: "
echo "./debug_openmm.sh | tee -a debug.log"
echo "Then send us debug.log"

set -euo pipefail

date

which -a python

conda info -a || echo "no conda"
mamba info -a || echo "no mamba"
micromamba info || echo "no micromamba"

nvidia-smi || echo "no nvidia-smi, are you on a gpu node?"

echo "test openmm"
python -m openmm.testInstallation || echo "testing openmm"

echo "checking plugin load failures"
python -c "import openmm; print(openmm.Platform.getPluginLoadFailures())" || echo "plugin load failures"

conda list || echo "no conda"
mamba list || echo "no mamba"
micromamba list || echo "no micromamba"
