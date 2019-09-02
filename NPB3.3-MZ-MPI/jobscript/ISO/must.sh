#!/bin/bash
# run this job from ./bin directory with "sh must.sh"

# benchmark configuration
PROCS=4
CLASS=W
EXE=./bt-mz_$CLASS.$PROCS

NEXUS="mustrun --must:mpiexec"

set -x

OMP_NUM_THREADS=4  $NEXUS  mpiexec -np $PROCS  $EXE

