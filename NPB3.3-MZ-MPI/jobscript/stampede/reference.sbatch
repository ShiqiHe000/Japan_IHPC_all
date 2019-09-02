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

# run the application
ibrun tacc_affinity $EXE
