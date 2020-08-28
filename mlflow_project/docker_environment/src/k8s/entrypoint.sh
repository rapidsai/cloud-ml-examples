#!/bin/bash
set -e
set -x
source activate rapids

ARGS=( "$@" )
echo "${ARGS[@]}"
echo "Calling: 'python ${ARGS[@]}'"
python --version
echo "---- ENV ----"
env
echo "---- LOCAL DIR ----"
ls -lah
python ${ARGS[@]}
echo "Python call returned: $?" 
