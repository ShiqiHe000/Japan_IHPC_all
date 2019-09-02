#!/bin/bash --login
# submit from ./bin.tau directory with "sbatch tau.sbatch"
#SBATCH --job-name="mzmpibt"
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=00:10:00
#SBATCH --res=sschool

# benchmark configuration
export OMP_NUM_THREADS=4
PROCS=16
CLASS=B
EXE=./bt-mz_$CLASS.$PROCS

# TAU configuration
#export TAU_COMM_MATRIX=1
#export TAU_THROTTLE=0
#export TAU_CALLPATH=1
#export TAU_CALLPATH_DEPTH=50

set -x
aprun -n $PROCS -d $OMP_NUM_THREADS  $EXE

