#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <inttypes.h>
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

void compute( double ** matrix_a, double ** matrix_b, double ** matrix_c, unsigned int matrix_size, unsigned int block_size )
{
  // initialize clocks
  struct timeval begin, end;
  int i, j, k, i_block, j_block, k_block;

  gettimeofday( &begin, NULL );
  // actual computation
  for ( i = 0; i < matrix_size; i += block_size ) {
    for ( k = 0; k < matrix_size; k += block_size ) {
      for ( j = 0; j < matrix_size; j += block_size ) {
        for ( i_block = i; i_block < i + block_size; i_block++ ) {
          for ( k_block = k; k_block < k + block_size; k_block++ ) {
            double tmp = (*matrix_a)[i_block * matrix_size + k_block];
            for ( j_block = j; j_block < j + block_size; j_block++ ) {
              (*matrix_c)[i_block * matrix_size + j_block] = (*matrix_c)[i_block * matrix_size + j_block] + tmp * (*matrix_b)[k_block * matrix_size + j_block];
            }
          }
        }
      }
    }
  }
  gettimeofday( &end, NULL );
  double elapsed = ( end.tv_sec - begin.tv_sec ) + ( ( end.tv_usec - begin.tv_usec ) / 1000000.0 );
  printf( "Matrix matrix multiplication: %.2lfs\n", elapsed );
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
  if( argc > 3 ) {
    printf( "Too many arguments supplied.\n" );
    exit( 0 );
  }
  if( argc == 1 || argc == 2 ) {
    printf( "Usage: ./mm <matrix_size> <block_size>\n" );
    exit( 0 );
  }
  if( ( atoi( argv[1] ) % atoi( argv[2] ) ) != 0 ) {
    printf( "<matrix_size> should be divisible by <block_size>\n" );
    exit( 0 );
  }
  unsigned int matrix_size = atoi( argv[1] );
  unsigned int block_size = atoi( argv[2] );

  printf( "Square matrix multiplication AxB with size: %d\n", matrix_size );

  double * matrix_a;
  double * matrix_b;
  double * matrix_c;

  allocate( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  assign( &matrix_a, &matrix_b, &matrix_c, matrix_size );
  compute( &matrix_a, &matrix_b, &matrix_c, matrix_size, block_size );
  check_results( &matrix_a, &matrix_c, matrix_size );
  free_memory( &matrix_a, &matrix_b, &matrix_c, matrix_size );

  return 0;
}
