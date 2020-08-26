#!/bin/bash
source activate rapids

ARGS=( "$@" )
echo "${ARGS[@]}"
echo "Calling: 'python ${ARGS[@]}'"
python "${ARG_SLICE[@]}"
