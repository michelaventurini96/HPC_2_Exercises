#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define RIGHT(proc, nproc) (proc - 1 + nproc) % nproc
#define LEFT(proc, nproc) (proc + 1 + nproc) % nproc
#define N 10

void swapPointers(int** const a, int** const b){
    int* t = *a;
    *a = *b;
    *b = t;
}

void fillVector(int* vec, const int val, const size_t n){
  for (size_t i = 0; i < n; i++)
    vec[i] = val;
}

void sumVector(int* const vec1, const int* const vec2, const size_t n){
  for (size_t i = 0; i < n; i++)
    vec1[i]+=vec2[i];
}

void printVector(const int* const vect, const size_t n){
  for (size_t i = 0; i < n; i++)
    printf("%d ", vect[i]);
  printf("\n");
}

int main(int argc, char * argv[])
{
  int rank, npes; // identifier of the process and total number of processes
  double startwtime, endwtime;
  
  MPI_Init( &argc, &argv ); // initialize session

  MPI_Request request;
  MPI_Status status;

  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // rank store the MPI identifier of the process
  MPI_Comm_size( MPI_COMM_WORLD, &npes ); // npes store the number of MPI processes

  // allocate and initialize buffers
  int* rec_buf = (int*) malloc(N*sizeof(int)); // buffer for receiving messages
  int* send_buf = (int*) malloc(N*sizeof(int)); // buffer for sending messages
  int* sum = (int*) malloc(N*sizeof(int)); // buffer to record the sum
  
  fillVector(rec_buf, 0, N);
  fillVector(send_buf, rank, N);
  fillVector(sum, 0, N);

  // each MPI process send to left and receive from right process
  for (int i = 0; i < npes; i++) {
    MPI_Isend(send_buf, N, MPI_INT, LEFT(rank, npes), 101, MPI_COMM_WORLD, &request);
    sumVector(sum, send_buf, N);
    MPI_Recv(rec_buf, N, MPI_INT, RIGHT(rank, npes), 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&request, &status);
    swapPointers(&rec_buf,&send_buf);
  }

  const int expected = (npes)*(npes-1)/2; // variable to check correctness of output
  printf("Rank = %d, Sum (Should be %d)\n", rank, expected);
  printVector(sum, N);

  MPI_Finalize(); // end MPI session

  // free memory
  free(rec_buf);
  free(send_buf);
  free(sum);

  return 0;
}
