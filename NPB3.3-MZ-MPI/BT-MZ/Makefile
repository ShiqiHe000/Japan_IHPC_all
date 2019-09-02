SHELL=/bin/sh
BENCHMARK=bt-mz
BENCHMARKU=BT-MZ
VEC=

include ../config/make.def

MAIN = bt_epik
MAIN = bt_scorep
MAIN = bt_scorep_loop
MAIN = bt_scorep_user
#MAIN = bt
Z_SOLVE = z_solve$(VEC)_scorep_user
#Z_SOLVE = z_solve$(VEC)

OBJS = $(MAIN).o  initialize.o exact_solution.o exact_rhs.o \
       set_constants.o adi.o  rhs.o zone_setup.o  \
       x_solve$(VEC).o y_solve$(VEC).o  exch_qbc.o solve_subs.o \
       $(Z_SOLVE).o add.o error.o verify.o mpi_setup.o \
       ${COMMON}/print_results.o ${COMMON}/timers.o

include ../sys/make.common

# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect).

# header.h: npbparams.h

${PROGRAM}: config
	@if [ x$(VERSION) = xvec ] ; then	\
		${MAKE} VEC=_vec exec;		\
	elif [ x$(VERSION) = xVEC ] ; then	\
		${MAKE} VEC=_vec exec;		\
	else					\
		${MAKE} exec;			\
	fi

exec: $(OBJS)
	${FLINK} ${FLINKFLAGS} -o ${PROGRAM} ${OBJS} ${F_LIB}

.F.o:
	${FCOMPILE} $<
.f.o:
	${FCOMPILE} $<

bt_epik.o:        bt_epik.F  mpi_stuff.h header.h npbparams.h
bt_scorep.o:      bt_scorep.F  mpi_stuff.h header.h npbparams.h
bt_scorep_loop.o: bt_scorep_loop.F  mpi_stuff.h header.h npbparams.h
bt_scorep_user.o: bt_scorep_user.F  mpi_stuff.h header.h npbparams.h
z_solve$(VEC)_scorep_user.o: z_solve$(VEC)_scorep_user.F  header.h work_lhs$(VEC).h npbparams.h

bt.o:             bt.f  mpi_stuff.h header.h npbparams.h
initialize.o:     initialize.f  header.h npbparams.h
exact_solution.o: exact_solution.f  header.h npbparams.h
exact_rhs.o:      exact_rhs.f  header.h npbparams.h
set_constants.o:  set_constants.f  header.h npbparams.h
adi.o:            adi.f  header.h npbparams.h
rhs.o:            rhs.f  header.h npbparams.h
zone_setup.o:     zone_setup.f mpi_stuff.h header.h npbparams.h
x_solve$(VEC).o:  x_solve$(VEC).f  header.h work_lhs$(VEC).h npbparams.h
y_solve$(VEC).o:  y_solve$(VEC).f  header.h work_lhs$(VEC).h npbparams.h
z_solve$(VEC).o:  z_solve$(VEC).f  header.h work_lhs$(VEC).h npbparams.h
solve_subs.o:     solve_subs.f  npbparams.h
add.o:            add.f  header.h npbparams.h
error.o:          error.f  header.h npbparams.h
verify.o:         verify.f  mpi_stuff.h header.h npbparams.h
exch_qbc.o:       exch_qbc.f mpi_stuff.h header.h npbparams.h
mpi_setup.o:      mpi_setup.f  mpi_stuff.h header.h npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
