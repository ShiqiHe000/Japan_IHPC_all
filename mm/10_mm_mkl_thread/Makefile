COMP=gcc
COMP_FLAGS=-std=c99 -O3 -Wpedantic -Wall -march=native
#COMP_FLAGS=-std=c99 -O3 -Wpedantic -Wall -DDEBUG -g -march=native
INC=-I${MKLROOT}/include
LIB=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -liomp5 -lm -ldl

mm: mm.o
	${COMP} ${COMP_FLAGS} mm.o -o mm ${INC} ${LIB}

mm.o: mm.c
	${COMP} ${COMP_FLAGS} -c mm.c ${INC}

clean:
	rm -rf *.o mm

