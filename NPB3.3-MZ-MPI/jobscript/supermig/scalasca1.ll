#!/bin/bash
#
# submit from ./bin.scalasca directory with "llsubmit scalasca1.ll"
# hybrid MPI+OpenMP job script for SuperMIG (fat nodes) IBM MPI
#
#@ job_type = parallel
#@ class = test
#@ node = 1
#@ total_tasks = 4
### other example 
##@ tasks_per_node = 4
#@ wall_clock_limit = 0:10:00
#@ job_name = scan_bt-mz
#@ network.MPI = sn_all,not_shared,us
#@ output = job$(jobid).out
#@ error = job$(jobid).out
##@ notification=always
##@ notify_user=erika.mustermann@xyz.de
#@ queue
. /etc/profile
. /etc/profile.d/modules.sh

module load scalasca/1.4.2

set -x
export MP_SINGLE_THREAD=no
export OMP_NUM_THREADS=4
# Pinning
export MP_TASK_AFFINITY=core:$OMP_NUM_THREADS

# Application load balancing
export NPB_MZ_BLOAD=0

# Scalasca configuration
#export EPK_FILTER=../config/scan.filt
#export EPK_METRICS=PAPI_TOT_CYC:PAPI_TOT_INS:PAPI_FP_INS
#export SCAN_ANALYZE_OPTS="-is"

NEXUS="scalasca -analyze -s"
$NEXUS  mpiexec -n 4 ./bt-mz_B.4


