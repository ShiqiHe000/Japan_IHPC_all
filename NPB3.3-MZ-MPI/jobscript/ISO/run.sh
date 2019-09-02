#!/bin/bash
# run from ./bin directory with "sh run.sh"

# benchmark configuration
PROCS=4
CLASS=W
EXE=./bt-mz_$CLASS.$PROCS

set -x

OMP_NUM_THREADS=4  mpiexec -np $PROCS  $EXE

