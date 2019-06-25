#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define N 10

void initVec(int* vec, size_t n){
  for (size_t i = 0; i < n; i++) {
    vec[i] = 0;
  }
}

void copy_vec(int* vec1, int* vec2, size_t n){
  for (size_t i = 0; i < n; i++) {
    vec1[i] = vec2[i];
  }
}

void sum_vect(int* vec1, int* vec2, size_t n){
  for (size_t i = 0; i < n; i++) {
    vec1[i]+=vec2[i];
  }
  //return vec1;
}

void print_vect(int* vect, size_t n){
  for (size_t i = 0; i < n; i++) {
    printf("%d ", vect[i]);
  }
  printf("\n");
}

int main(int argc, char * argv[])
{
  //int N = 10;
  int rank, npes;
  double startwtime, endwtime;
  MPI_Init( &argc, &argv );

  MPI_Request request;
  MPI_Status status;

  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // rank store the MPI identifier of the process
  MPI_Comm_size( MPI_COMM_WORLD, &npes ); // npes store the number of MPI processes

  //Synchronize all processes and get the begin time
  MPI_Barrier(MPI_COMM_WORLD);
  startwtime = MPI_Wtime();

  int* local_buf = (int*) malloc(N*sizeof(int));
  for (size_t i = 0; i < N; i++) local_buf[i] = 0;

  int* rec_buf = (int*) malloc(N*sizeof(int));
  //int* rec_buf = (int*) calloc(N,sizeof(int));
  //initVec(rec_buf, N);

  int* rank_v = (int*) malloc(N*sizeof(int));
  for (size_t i = 0; i < N; i++) rank_v[i] = rank;

  int right = (rank + 1) % npes;
  int left = rank - 1;
  if (left < 0) left = npes - 1;

  for (int i = 0; i < npes; i++) {
    MPI_Isend(&local_buf[0], N, MPI_INT, left, 101, MPI_COMM_WORLD,&request);
    MPI_Recv(&rec_buf[0], N, MPI_INT, right, 101, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    sum_vect(rec_buf, rank_v, N);
    copy_vec(local_buf, rec_buf, N);//local_buf=sum_vect(rec_buf, rank_v, N);
    initVec(rec_buf, N);
    MPI_Wait(&request, &status);
  }

  //Synchronize all processes and get the end time
  MPI_Barrier(MPI_COMM_WORLD);
  endwtime = MPI_Wtime();

  int expected = (npes)*(npes-1)/2;

  printf("Rank = %d, Sum (Should be %d)\n", rank, expected);
  print_vect(local_buf, N);

  if (rank==0) {
    printf("Sum (Should be %d)\n", expected);
    print_vect(local_buf, N);
    printf("Time of execution: %f\n", endwtime-startwtime);
}

  MPI_Finalize();

  return 0;
} //end main
