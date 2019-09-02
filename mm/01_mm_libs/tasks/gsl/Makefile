COMP=gcc
COMP_FLAGS=-O3 -Wpedantic -Wall -march=native
#COMP_FLAGS=-O3 -Wpedantic -Wall -DDEBUG -g -march=native
INC=-I${EBROOTGSL}/include
LIB=-L${EBROOTGSL}/lib -lgsl -lgslcblas -lm

mm: mm.o
	${COMP} ${COMP_FLAGS} mm.o -o mm ${INC} ${LIB}

mm.o: mm.c
	${COMP} ${COMP_FLAGS} -c mm.c ${INC}

clean:
	rm -rf *.o mm

