#!/bin/bash
#SBATCH -J mzmpibt          # Job name
#SBATCH -o mzmpibt.o%j      # Name of stdout output file(%j expands to jobId)
#SBATCH -e mzmpibt.e%j      # Name of stderr output file(%j expands to jobId)
#SBATCH -p serial           # Serial queue for serial and OpenMP jobs
#SBATCH -N 1                # Total number of nodes requested (16 cores/node)
#SBATCH -n 4                # Total number of mpi tasks requested
#SBATCH -t 00:05:00         # Run time (hh:mm:ss) - 5 minutes

# benchmark configuration
export OMP_NUM_THREADS=4
PROCS=4
CLASS=B
EXE=./bt-mz_$CLASS.$PROCS

# prevent scan from taking tacc_affinity as target
export SCAN_TARGET=$EXE
# tell scan the number of MPI processes as this can't be figured 
# out via the ibrun parameters
export SCAN_MPI_RANKS=$PROCS
# run the application
scan ibrun tacc_affinity $EXE
#scan -f scorep.filt ibrun tacc_affinity $EXE
