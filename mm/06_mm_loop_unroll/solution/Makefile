COMP=gcc
COMP_FLAGS=-O2 -Wpedantic -Wall -g -march=native 
#COMP_FLAGS=-O2 -Wpedantic -Wall -g -march=native -funroll-all-loops 
#COMP_FLAGS=-O2 -Wpedantic -Wall -g -march=native -funroll-loops
#COMP_FLAGS=-O2 -Wpedantic -Wall -g -DDEBUG

mm: mm.o
	${COMP} ${COMP_FLAGS} mm.o -o mm

mm.o: mm.c
	${COMP} ${COMP_FLAGS} -c mm.c

clean:
	rm -rf *.o mm

