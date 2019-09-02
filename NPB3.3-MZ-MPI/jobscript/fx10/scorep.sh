#!/bin/bash
# run from ./bin.scorep directory with "pjsub scorep.sh"
#PJM -j
#PJM --mpi proc=8
#PJM --rsc-list node=2
#PJM --rsc-list elapse=00:10:00
#PJM --rsc-list rscgrp=small

#source /work/system/Env_base

# benchmark configuration
PROCS=8
CLASS=B

EXE=./bt-mz_$CLASS.$PROCS

export OMP_NUM_THREADS=4
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

set -x

# Score-P measurement configuration
#export SCOREP_EXPERIMENT_DIRECTORY=scorep_sum
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_FP_INS
#export SCOREP_METRIC_PAPI_PER_PROCESS=PAPI_L2_TCM
#export SCOREP_METRIC_RUSAGE=ru_stime
#export SCOREP_METRIC_RUSAGE_PER_PROCESS=ru_maxrss

mpiexec  $EXE

