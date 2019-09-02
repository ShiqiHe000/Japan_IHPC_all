#!/bin/bash
#
# submit from ./bin directory with "llsubmit scorep.ll"
# hybrid MPI+OpenMP job script for SuperMUC (thin nodes) IBM MPI
#
#@ job_type = parallel
#@ class = test
#@ class = general
#@ island_count = 1
#@ node = 1
#@ total_tasks = 4
### other example 
##@ tasks_per_node = 4
#@ wall_clock_limit = 00:10:00
#@ job_name = run_bt-mz
#@ network.MPI = sn_all,not_shared,us
#@ output = job$(jobid).out
#@ error = job$(jobid).out
##@ notification=always
##@ notify_user=erika.mustermann@xyz.de
#@ queue
. /etc/profile
. /etc/profile.d/modules.sh

export MP_SINGLE_THREAD=no
export OMP_NUM_THREADS=4

# Pinning
export MP_TASK_AFFINITY=core:$OMP_NUM_THREADS

# Application load balancing
export NPB_MZ_BLOAD=0

# Score-P configuration
#export SCOREP_EXPERIMENT_DIRECTORY=scorep_experiment
#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_METRIC_RUSAGE=all
#export SCOREP_METRIC_PAPI=PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_FP_INS
#export SCOREP_ENABLE_TRACING=true

mpiexec -n 4 ./bt-mz_B.4
