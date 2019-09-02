#!/bin/bash
# submit from ./bin directory with "sbatch reference.sbatch"
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
#SBATCH --output=mzmpibt_%j.out
#SBATCH --error=mzmpibt_%j.err
#SBATCH --mail-user=a.strube@fz-juelich.de
#SBATCH --mail-type=ALL
module load intel impi scorep scalasca cube

# Benchmark configuration (disable load balancing with threads)
export NPB_MZ_BLOAD=0
PROCS=32
CLASS=C

# Run the application
srun ./bt-mz_$CLASS.$PROCS
