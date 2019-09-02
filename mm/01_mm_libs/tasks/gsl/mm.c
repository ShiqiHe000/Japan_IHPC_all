#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <float.h>
#include <math.h>
#include <gsl/gsl_blas.h>

void print_matrix( unsigned int matrix_size, gsl_matrix ** input_matrix, char* matrix_name )
{
  int i, j;
  printf ( "\nInput Matrix %s:", matrix_name );
  for ( i = 0; i < matrix_size; i++ ) {
    printf ( "\n\t" );
    for ( j = 0; j < matrix_size; j++ )
      printf ( "%.2lf ", gsl_matrix_get ( *input_matrix, i, j ) );
  }
  printf("\n");
}

void allocate( gsl_matrix ** matrix_a, gsl_matrix ** matrix_b, gsl_matrix ** matrix_c, unsigned int matrix_size )
{
  // initialize clocks
  struct timeval begin, end;

  gettimeofday( &begin, NULL );
  // allocate memory
  *matrix_a = gsl_matrix_alloc( matrix_size, matrix_size );
  *matrix_b = gsl_matrix_alloc( matrix_size, matrix_size );
  *matrix_c = gsl_matrix_alloc( matrix_size, matrix_size );
  gettimeofday( &end, NULL );
  double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
  printf( "Memory allocation: %.2lfs\n", elapsed );
}

void assign( gsl_matrix ** matrix_a, gsl_matrix ** matrix_b, gsl_matrix ** matrix_c, unsigned int matrix_size )
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
      gsl_matrix_set( *matrix_a, i, j, ( double ) ( rand() % 100) );
      gsl_matrix_set( *matrix_b, i, j, ( ( i == j ) ? 1.0 : 0.0 ) );
      gsl_matrix_set( *matrix_c, i, j, 0.0 );
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

void compute( gsl_matrix ** matrix_a, gsl_matrix ** matrix_b, gsl_matrix ** matrix_c, unsigned int matrix_size, unsigned int repetitions )
{
  int k;
  double time_sum = 0.0; double time_min = DBL_MAX;
  for ( k = 0; k < repetitions; k++ ){
    // initialize clocks
    struct timeval begin, end;
//	gettimeofday()
    // TODO: implement matrix-matrix multiplication with GSL library and measure time
    int gsl_blas_dgemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, double alpha, const gsl_matrix * A, const gsl_matrix * B, double beta, gsl_matrix * C)
    // These functions compute the matrix-matrix product and sum C = \alpha op(A) op(B) + \beta C where op(A) = A, A^T, A^H for TransA = CblasNoTrans, CblasTrans, CblasConjTrans and similarly for the parameter TransB.
    // See: https://www.gnu.org/software/gsl/doc/html/blas.html

    printf( "# %d matrix-matrix multiplication: %.2lfs\n", k, elapsed );
    time_sum += elapsed;
    time_min = ( time_min < elapsed ) ? time_min : elapsed;
  }
  printf( "AVG time: %.2lfs MIN time: %.2lfs\n", time_sum/repetitions, time_min );
#ifdef DEBUG
  print_matrix( matrix_size, matrix_c, "C" );
#endif
}

void check_results( gsl_matrix ** matrix_a, gsl_matrix ** matrix_c, unsigned int matrix_size )
{
  // Compare results
  int i, j;
  double abs_error, max_abs_error = 0.0, sum_abs_error = 0.0;
  for ( i = 0; i < matrix_size; i++ ){
    for ( j = 0; j < matrix_size; j++ ) {

      abs_error = fabs ( gsl_matrix_get ( *matrix_c, i, j ) - gsl_matrix_get ( *matrix_a, i, j ) );
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


void free_memory( gsl_matrix ** matrix_a, gsl_matrix ** matrix_b, gsl_matrix ** matrix_c, unsigned int matrix_size )
{
  // initialize clocks
  struct timeval begin, end;
  gettimeofday( &begin, NULL );
  gsl_matrix_free( *matrix_a );
  gsl_matrix_free( *matrix_b );
  gsl_matrix_free( *matrix_c );
  gettimeofday( &end, NULL );
  double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
  printf( "Free memory: %.2lfs\n", elapsed );
}

int main(int argc, char *argv[])
{

  if( argc > 3 ) {
    printf( "Too many arguments supplied.\n" );
    exit( 0 );
  }
  if( argc == 1 ) {
    printf( "Usage: ./mm <matrix_size> [<repetitions>]\n" );
    exit( 0 );
  }
  unsigned int matrix_size = atoi( argv[1] );
  unsigned int repetitions = ( argv[2] == NULL ) ? 1 : atoi( argv[2] );
  printf( "Square matrix multiplication AxB with size: %d Repetitions: %d\n", matrix_size, repetitions );
  gsl_matrix * matrix_a;
  gsl_matrix * matrix_b;
  gsl_matrix * matrix_c;

  allocate( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  assign( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  compute( &matrix_a, &matrix_b, &matrix_c, matrix_size, repetitions );
  check_results( &matrix_a, &matrix_c, matrix_size );
  free_memory( &matrix_a, &matrix_b, &matrix_c, matrix_size );

  return 0;
}
