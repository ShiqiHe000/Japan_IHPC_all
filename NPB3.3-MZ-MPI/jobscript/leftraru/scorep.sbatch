#!/bin/bash
# submit from ./bin.scorep directory with "sbatch scorep_mic.sbatch"
# Leftraru has 4 partitions: 
# - slims: 128 slim nodes
# - fats: 4 fat nodes + mic in offload mode
# - mics: partition with the 12 Xeon Phi
# - all: all slim+fat nodes
#SBATCH --job-name=mzmpibt-scorep
#SBATCH --exclusive=user # Doesnt allow node sharing
#SBATCH --partition=slims
#SBATCH --ntasks-per-socket=8 # Maximum is 10
#SBATCH -n 32  # Ideally is a multiple of 20 or of ntasks-per-socket
#SBATCH --output=mzmpibt_scorep-%j.out
#SBATCH --error=mzmpibt_scorep-%j.err
#SBATCH --mail-user=a.strube@fz-juelich.de
#SBATCH --mail-type=ALL
module load intel impi scorep cube scalasca

# Benchmark configuration (disable load balancing with threads)
export NPB_MZ_BLOAD=0
PROCS=32
CLASS=C

# Score-P measurement configuration
#export SCOREP_EXPERIMENT_DIRECTORY=scorep_bt-mz_sum
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=14M
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_TOT_CYC
#export SCOREP_ENABLE_TRACING=true

# Run the application
srun ./bt-mz_${CLASS}.${PROCS}
