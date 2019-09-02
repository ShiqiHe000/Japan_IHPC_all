#!/bin/bash
# submit from ./bin directory with "sbatch reference.sbatch"
# Leftraru has 4 partitions: 
# - slims: 128 slim nodes
# - fats: 4 fat nodes + mic in offload mode
# - mics: partition with the 12 Xeon Phi
# - all: all slim+fat nodes
#SBATCH --job-name=mzmpibt-scalasca-profile
#SBATCH --partition=slims
#SBATCH --exclusive=user # Doesnt allow node sharing
#SBATCH --partition=slims
#SBATCH --ntasks-per-socket=8 # Maximum is 10
#SBATCH -n 32  # Ideally is a multiple of 20 or of ntasks-per-socket
#SBATCH --output=mzmpibt-scalasca-profile_%j.out
#SBATCH --error=mzmpibt-scalasca-profile_%j.err
#SBATCH --mail-user=a.strube@fz-juelich.de
#SBATCH --mail-type=ALL
module load intel impi scorep scalasca cube

# Benchmark configuration (disable load balancing with threads)
export NPB_MZ_BLOAD=0
PROCS=32
CLASS=C
EXE=./bt-mz_$CLASS.$PROCS

#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=14M

# prevent scan from taking srun as target
export SCAN_TARGET=$EXE
# tell scan the number of MPI processes as this can't be figured 
# out via the ibrun parameters
export SCAN_MPI_RANKS=$PROCS
# run the application
scalasca -analyze -s srun $EXE
