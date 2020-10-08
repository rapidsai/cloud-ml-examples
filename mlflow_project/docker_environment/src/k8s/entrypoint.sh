#!/bin/bash
set -e
set -x
source activate rapids
nvidia-smi

ARGS=( "$@" )
python --version
echo "Calling: 'python ${ARGS[@]}'"
python ${ARGS[@]}
echo "Python call returned: $?" 
