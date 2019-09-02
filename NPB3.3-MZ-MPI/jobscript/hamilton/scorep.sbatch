#!/bin/bash --login
# submit from ./bin.scorep directory with "sbatch scorep.sbatch"
#SBATCH --job-name="mzmpibt"
#SBATCH --output=scorep-%A.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --partition=par6.q

# benchmark configuration
export NPB_MZ_BLOAD=0
export OMP_NUM_THREADS=3
PROCS=8
CLASS=C
EXE=./bt-mz_$CLASS.$PROCS

# Score-P measurement configuration
#export SCOREP_EXPERIMENT_DIRECTORY=scorep_sum
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_FP_INS
#export SCOREP_METRIC_PAPI_PER_PROCESS=PAPI_L2_TCM
#export SCOREP_METRIC_RUSAGE=ru_stime
#export SCOREP_METRIC_RUSAGE_PER_PROCESS=ru_maxrss
#export SCOREP_TOTAL_MEMORY=50M

mpirun -np $PROCS  $EXE

