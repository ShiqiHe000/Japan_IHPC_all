COMP=gcc
COMP_FLAGS=-O3 -pg -Wpedantic -Wall -march=native
#COMP_FLAGS=-O3 -pg -Wpedantic -Wall -fno-inline -fno-reorder-functions -march=native

mm: mm.o
	${COMP} ${COMP_FLAGS} mm.o -o mm

mm.o: mm.c
	${COMP} ${COMP_FLAGS} -c mm.c

clean:
	rm -rf *.o mm

