#!/bin/sh
### submit from ./bin.scorep directory with "pjsub scalasca2.sh"
#PJM -j
#PJM --mpi proc=8
#PJM --rsc-list node=2
#PJM --rsc-list elapse=00:10:00
#PJM --rsc-list rscgrp=small

#source /work/system/Env_base
#module load UNITE scalasca/2.1
export PATH=/home/vi-hps/scalasca/2.1/bin:$PATH

# benchmark configuration
CLASS=B
NPROCS=8
EXE=./bt-mz_${CLASS}.${NPROCS}

export OMP_NUM_THREADS=4
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

# Scalasca2/Score-P configuration
export SCAN_MPI_REDIRECT=true
export SCAN_ANALYZE_OPTS="--time-correct"

export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=50M
#export SCOREP_METRIC_PAPI=PAPI_FP_OPS

NEXUS="scalasca -analyze"

set -x

$NEXUS  mpiexec $EXE
