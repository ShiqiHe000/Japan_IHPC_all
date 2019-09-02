#!/bin/bash
# run from bin.scorep directory with "sh scalasca2.sh"

# benchmark configuration
PROCS=4
CLASS=W
EXE=./bt-mz_$CLASS.$PROCS

set -x

# Scalasca2 configuration
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=26M

NEXUS="scalasca -analyze"
OMP_NUM_THREADS=4  $NEXUS  mpiexec -np $PROCS  $EXE


