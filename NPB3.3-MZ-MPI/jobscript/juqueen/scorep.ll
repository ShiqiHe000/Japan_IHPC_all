#
# Sample Juqueen hybrid bt-mz jobscript for Score-P measurement
#
# @ job_name = scorep.bt-mz
# @ comment = "Score-P bt-mz"
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL
# @ wall_clock_limit = 00:30:00
# @ notification = error
# @ notify_user = $(user)@fz-juelich.de
# @ job_type = bluegene
# @ bg_size = 32
# @ queue

CLASS=C
NPROCS=32

EXE=./bt-mz_${CLASS}.${NPROCS}

export OMP_NUM_THREADS=8
export NPB_MZ_BLOAD=0 # disable load balancing with dynamic threads

module load UNITE scorep/1.2.1

#export SCOREP_FILTERING_FILE=../config/scorep.filt
#export SCOREP_TOTAL_MEMORY=30M

    runjob --np $NPROCS --exp-env NPB_MZ_BLOAD --exp-env OMP_NUM_THREADS --exe $EXE
