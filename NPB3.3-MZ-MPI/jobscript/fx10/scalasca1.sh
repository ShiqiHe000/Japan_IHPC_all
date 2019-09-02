#!/bin/sh
### submit from ./bin.scalasca directory with "pjsub scalasca1.sh"
#PJM -j
#PJM --mpi proc=8
#PJM --rsc-list node=2
#PJM --rsc-list elapse=00:10:00
#PJM --rsc-list rscgrp=small

#source /work/system/Env_base
#module load UNITE scalasca/1.4.4
export PATH=/home/vi-hps/scalasca/1.4.4/bin:$PATH

# benchmark configuration
CLASS=B
NPROCS=8

EXE=./bt-mz_${CLASS}.${NPROCS}

export OMP_NUM_THREADS=4
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

# Scalasca1 configuration
export SCAN_MPI_REDIRECT=true
export SCAN_ANALYZE_OPTS="-is"

export EPK_TOFU_DIM_X=1
export EPK_TOFU_DIM_Y=1
export EPK_FILTER=../config/scan.filt

NEXUS="scalasca -analyze"

set -x

$NEXUS  mpiexec  $EXE
