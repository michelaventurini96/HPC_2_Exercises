#include <stdlib.h>
#include <stdio.h>

void print_usage( int * a, int N, int nthreads ) {

  int tid, i;
  for( tid = 0; tid < nthreads; ++tid ) {

    fprintf( stdout, "%d: ", tid );

    for( i = 0; i < N; ++i ) {
      if( a[ i ] == tid) fprintf( stdout, "*" );
      else fprintf( stdout, " ");
    }
    printf("\n");
  }
}

int main(){

  
  const int N = 250;
  int a[N];

  //SERIAL
  int thread_id = 0;
  int nthreads = 1;

  for(int i = 0; i < N; ++i) {
    a[i] = thread_id;
  }

  print_usage(a, N, nthreads);
  
  //STATIC
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int i = 0;
    #pragma omp for schedule(static) private(i)
    for(i = 0; i < N; ++i) a[i] = thread_id;

    #pragma omp single
    print_usage(a, N, nthreads);
  }

  //STATIC 1
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int i = 0;
    #pragma omp for schedule(static,1) private(i)
    for(i = 0; i < N; ++i) a[i] = thread_id;

    #pragma omp single
    print_usage(a, N, nthreads);
  }

  //STATIC 10
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int i = 0;
    #pragma omp for schedule(static,10) private(i)
    for(i = 0; i < N; ++i) a[i] = thread_id;

    #pragma omp single
    print_usage(a, N, nthreads);
  }

  //DYNAMIC 1
  #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int nthreads = omp_get_num_threads();

      int i = 0;
      #pragma omp for schedule(dynamic) private(i)
      for(i = 0; i < N; ++i) a[i] = thread_id;

      #pragma omp single
      print_usage(a, N, nthreads);
    }
  
  //DYNAMIC 10
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int i = 0;
    #pragma omp for schedule(dynamic,10) private(i)
    for(i = 0; i < N; ++i) a[i] = thread_id;

    #pragma omp single
    print_usage(a, N, nthreads);
  }


}
