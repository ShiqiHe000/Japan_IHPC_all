#!/bin/bash
# run from bin.scorep directory with "sh periscope.sh"

# benchmark configuration
PROCS=4
CLASS=W
EXE=./bt-mz_$CLASS.$PROCS

export OMP_NUM_THREADS=4

# Periscope configuration
export SCOREP_FILTERING_FILE=../config/scorep.filt

set -x

psc_frontend --apprun=$EXE --mpinumprocs=$PROCS --ompnumthreads=$OMP_NUM_THREADS --force-localhost --phase="ITERATION" --strategy=OMP


