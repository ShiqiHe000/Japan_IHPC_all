#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <float.h>
#include <math.h>

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

void clear_matrix( double ** matrix, unsigned int matrix_size )
{
  int i;
  for ( i = 0; i < matrix_size * matrix_size; i++ )
    (*matrix)[i] = 0.0;
}

void compute( double ** matrix_a, double ** matrix_b, double ** matrix_c, unsigned int matrix_size, unsigned int repetitions  )
{
  int m;
  double time_sum = 0.0; double time_min = DBL_MAX;
  for ( m = 0; m < repetitions; m++ ){
    // initialize clocks
    struct timeval begin, end;
    int i, j, k;

    if ( m > 0 )
      clear_matrix( matrix_c, matrix_size );
  
  gettimeofday( &begin, NULL );
  // actual computation
  #pragma omp parallel for
    for ( i = 0; i < matrix_size; i++ ) {
      int offset_i = i * matrix_size;
      for ( k = 0; k < matrix_size; k++ ) {
        int offset_k = k * matrix_size;
        double tmp = (*matrix_a)[offset_i + k];
        for ( j = 0; j < matrix_size; j++ ) {
          (*matrix_c)[offset_i + j] = (*matrix_c)[offset_i + j] + tmp * (*matrix_b)[offset_k + j];
        }
      }
    }
    gettimeofday( &end, NULL );
    double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
    printf( "# %d matrix-matrix multiplication: %.2lfs\n", m, elapsed );
    time_sum += elapsed;
    time_min = ( time_min < elapsed ) ? time_min : elapsed;
  }
  printf( "AVG time: %.2lfs MIN time: %.2lfs\n", time_sum/repetitions, time_min );

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
  unsigned int matrix_size = atoi( argv[1] );
  unsigned int repetitions = ( argv[2] == NULL ) ? 1 : atoi( argv[2] );
  if( argc > 3 ) {
    printf( "Too many arguments supplied.\n" );
    exit( 0 );
  }
  if( argc == 1 ) {
    printf( "Usage: ./mm <matrix_size> [<repetitions>]\n" );
    exit( 0 );
  }
  printf( "Square matrix multiplication AxB with size: %d Repetitions: %d\n", matrix_size, repetitions );

  double * matrix_a;
  double * matrix_b;
  double * matrix_c;

  allocate( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  assign( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  compute( &matrix_a, &matrix_b, &matrix_c, matrix_size, repetitions );
  check_results( &matrix_a, &matrix_c, matrix_size );
  free_memory( &matrix_a, &matrix_b, &matrix_c, matrix_size );

  return 0;
}
