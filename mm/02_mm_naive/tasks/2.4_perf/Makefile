COMP=gcc
COMP_FLAGS=-O3 -g -Wpedantic -Wall -march=native

mm: mm.o
	${COMP} ${COMP_FLAGS} mm.o -o mm

mm.o: mm.c
	${COMP} ${COMP_FLAGS} -c mm.c

clean:
	rm -rf *.o mm

