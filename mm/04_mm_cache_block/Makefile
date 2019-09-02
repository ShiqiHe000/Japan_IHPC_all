COMP=gcc
COMP_FLAGS=-O2 -Wpedantic -Wall
#COMP_FLAGS=-O2 -Wpedantic -Wall -fopenmp -ftree-vectorize -fopt-info-vec=vec.report -march=native -g -DDEBUG
#COMP_FLAGS=-O2 -Wpedantic -Wall -pg -fno-inline -fno-reorder-functions
#COMP_FLAGS=-O2 -Wpedantic -Wall -ftree-vectorize -fopt-info-vec-missed=vec.report

mm: mm.o
	${COMP} ${COMP_FLAGS} mm.o -o mm

mm.o: mm.c
	${COMP} ${COMP_FLAGS} -c mm.c

clean:
	rm -rf *.o mm

