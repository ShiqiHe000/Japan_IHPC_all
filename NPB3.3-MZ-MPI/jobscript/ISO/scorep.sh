#!/bin/bash
# run from ./bin.scorep directory with "sh scorep.sh"

# benchmark configuration
PROCS=4
CLASS=W
EXE=./bt-mz_$CLASS.$PROCS

set -x

# Score-P measurement configuration
#export SCOREP_EXPERIMENT_DIRECTORY=scorep_bt-mz_W_4x4_sum
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_FP_INS
#export SCOREP_METRIC_PAPI_PER_PROCESS=PAPI_L3_DCM
#export SCOREP_METRIC_RUSAGE=ru_stime
#export SCOREP_METRIC_RUSAGE_PER_PROCESS=ru_maxrss

OMP_NUM_THREADS=4  mpiexec -np $PROCS  $EXE

