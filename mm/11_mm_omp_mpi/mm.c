#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>

void print_matrix( unsigned int matrix_size, double **input_matrix, char* matrix_name )
{
  int i, j;
  printf ( "\nMatrix %s:", matrix_name );
  for ( i = 0; i < matrix_size; i++ ) {
    printf ( "\n\t" );
    for ( j = 0; j < matrix_size; j++ )
      printf ( "%.2lf ", (*input_matrix)[i * matrix_size + j] );
  }
  printf("\n");
}

void allocate( double ** matrix_a, double ** matrix_b, double ** matrix_c, unsigned int matrix_size )
{
  // initialize clocks
  struct timeval begin, end;

  gettimeofday( &begin, NULL );
  // allocate memory
  *matrix_a = malloc( matrix_size * matrix_size * sizeof( double * ) );
  *matrix_b = malloc( matrix_size * matrix_size * sizeof( double * ) );
  *matrix_c = malloc( matrix_size * matrix_size * sizeof( double * ) );
  gettimeofday( &end, NULL );
  double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
  printf( "Memory allocation: %.2lfs\n", elapsed );
}

void assign( double ** matrix_a, double ** matrix_b, double ** matrix_c, unsigned int matrix_size )
{
  // initialize clocks
  struct timeval begin, end;
  int i, j;
  // initialize random seed
  srand( 42 );

  gettimeofday( &begin, NULL );
  // assign values
  for ( i = 0; i < matrix_size; i++ ) {
    for ( j = 0; j < matrix_size; j++)  {
      (*matrix_a)[i * matrix_size + j] = ( double ) ( rand() % 100 );
      (*matrix_b)[i * matrix_size + j] = ( i == j ) ? 1.0 : 0.0;
      (*matrix_c)[i * matrix_size + j] = 0.0;
    }
  }
  gettimeofday( &end, NULL );
  double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
  printf( "Set matrix values: %.2lfs\n", elapsed );

#ifdef DEBUG
  print_matrix( matrix_size, matrix_a, "A" );
  print_matrix( matrix_size, matrix_b, "B" );
#endif
}

void compute( double ** matrix_a, double ** matrix_b, double ** matrix_c, unsigned int matrix_size, int rows )
{
  // initialize clocks
  int i, j, k;
  
  // actual computation
  #pragma omp parallel for
  for ( i = 0; i < rows; i++ ) {
    int offset_i = i * matrix_size;
    for ( k = 0; k < matrix_size; k++ ) {
      int offset_k = k * matrix_size;
      double tmp = (*matrix_a)[offset_i + k];
      for ( j = 0; j < matrix_size; j++ ) {
        (*matrix_c)[offset_i + j] = (*matrix_c)[offset_i + j] + tmp * (*matrix_b)[offset_k + j];
      }
    }
  }
#ifdef DEBUG
  print_matrix( matrix_size, matrix_c, "C" );
#endif
}

void free_memory( double ** matrix_a, double ** matrix_b, double ** matrix_c, unsigned int matrix_size )
{
  // initialize clocks
  struct timeval begin, end;

  gettimeofday( &begin, NULL );
  free( *matrix_a );
  free( *matrix_b );
  free( *matrix_c );
  gettimeofday( &end, NULL );
  double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
  printf( "Free memory: %.2lfs\n", elapsed );
}

void check_results( double ** matrix_a, double ** matrix_c, unsigned int matrix_size )
{
  // Compare results
  int i, j;
  double abs_error, max_abs_error = 0.0, sum_abs_error = 0.0;
  for ( i = 0; i < matrix_size; i++ ){
    for ( j = 0; j < matrix_size; j++ ) {

      abs_error = fabs ( (*matrix_c)[i * matrix_size + j] - (*matrix_a)[i * matrix_size + j]);
      sum_abs_error += abs_error;

      if (abs_error > max_abs_error)
        max_abs_error = abs_error;
    }
  }
#ifdef DEBUG
  printf ("maxAbsError: %4.4f, sumAbsError: %4.4f\n", max_abs_error, sum_abs_error);
#endif
  if (max_abs_error < 2.0e-5) {
    printf ("Program terminated SUCCESSFULLY\n");
  } else {
    printf ("--> Result not correct:  check your code\n");
  }
}

int main(int argc, char *argv[])
{
  int  size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  unsigned int matrix_size = atoi( argv[1] );
  if( rank == 0 ) {
    if( argc > 2 ) {
      printf( "Too many arguments supplied.\n" );
      exit( 0 );
    }
    if( argc == 1 ) {
      printf( "Usage: ./mm <matrix_size>\n" );
      exit( 0 );
    }
    if( ( atoi( argv[1] ) % size ) != 0 ) {
      printf( "Number of MPI processes should be divisible by <matrix_size>\n" );
      exit( 0 );
    }
    printf( "Square matrix multiplication AxB with size: %d\n", matrix_size );
    printf( "Values are double. Size for double: %lu bytes\n", sizeof( double ) );
  }

  double * matrix_a;
  double * matrix_b;
  double * matrix_c;
  if( rank == 0 ) {
    allocate( &matrix_a, &matrix_b, &matrix_c, matrix_size );
    assign( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  }

  unsigned int rows = matrix_size / size;
  double * local_matrix_a = malloc( rows * matrix_size * sizeof( double * ) );
  double * local_matrix_b = malloc( matrix_size * matrix_size * sizeof( double * ) );
  double * local_matrix_c = malloc( rows * matrix_size * sizeof( double * ) );
  if( rank == 0 )
    local_matrix_b = matrix_b;

  // scatter matrix A
  MPI_Scatter( &matrix_a[0], rows * matrix_size, MPI_DOUBLE, &local_matrix_a[0], rows * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  // broadcast complete matrix B
  MPI_Bcast( &local_matrix_b[0], matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  double start = MPI_Wtime();  
  compute( &local_matrix_a, &local_matrix_b, &local_matrix_c, matrix_size, rows );
  double end = MPI_Wtime();

  // collect result matrix
  MPI_Gather( &local_matrix_c[0], rows * matrix_size, MPI_DOUBLE, &matrix_c[0], rows * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  if( rank == 0 ) {
    printf( "Matrix matrix multiplication: %.2lfs\n", end - start );
    check_results( &matrix_a, &matrix_c, matrix_size );
    free_memory( &matrix_a, &local_matrix_b, &matrix_c, matrix_size );
  }
  free( local_matrix_a );
  free( local_matrix_c );  
  return 0;
}
