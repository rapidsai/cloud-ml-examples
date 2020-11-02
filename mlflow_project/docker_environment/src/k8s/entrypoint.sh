#!/bin/bash
# Activates the correct Anaconda environment, and runs the command passed to the container.

set -e
set -x
source activate rapids
nvidia-smi

ARGS=( "$@" )
python --version
echo "Calling: 'python ${ARGS[@]}'"
python ${ARGS[@]}
echo "Python call returned: $?" 
