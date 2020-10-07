#!/bin/bash
set -e
set -x
source activate rapids
nvidia-smi

ARGS=( "$@" )
python --version
echo "${ARGS[@]}"
echo "Calling: 'python ${ARGS[@]}'"
echo "---- ENV ----"
env
echo "---- LOCAL DIR ----"
ls -lah
python ${ARGS[@]}
echo "Python call returned: $?" 
