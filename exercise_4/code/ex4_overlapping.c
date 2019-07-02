#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

size_t globalIdx(const unsigned int n, const int npes, const int rank){
  int elems = (n*n)/npes;
  int rest = (n*n)%npes;
  
  size_t global_i = 0;

  for(int i=0; i<rank; i++)
    global_i+=(elems + (i<rest));

  return global_i;
}

void printPartialMatrix(const int* const matrix, const unsigned int n, const int rank, const int npes){
  size_t global_i = globalIdx(n, npes, rank);
  int rest = (n*n)%npes, elems = n*n/npes + (rank<rest), res;

  for (int i = 0; i < elems; i++) {
      printf("%d ", matrix[i]);
      res=(global_i+1)%n;
      if(res==0 && global_i!=0) printf("\n");
      global_i++;
  }
}

void printAllMatrix(const int* const matrix, const unsigned int n){
  int res;
  for (unsigned int i = 0; i < (n*n); i++) {
      printf("%d ", matrix[i]);
      res = (i+1)%n;
      if(res==0 && i!=0) printf("\n");
  }
}

void checkMatrix(const unsigned int n, const char* const fileName){
  int* m = (int*) calloc(sizeof(int),(n*n));
  FILE* m_file;
  m_file = fopen(fileName,"r");
  fread(m,sizeof(int),(n*n),m_file);
  printAllMatrix(m, n);
  free(m);
  fclose(m_file);
}

void initLocalMatrix(int *m, const int rank, const unsigned int local_elems, const unsigned int n, const int npes){

  int global_i = globalIdx(n, npes, rank);

  for (unsigned int i = 0; i < local_elems; i++){
    if (i+global_i==0 || (i+global_i)%(n+1)==0) m[i] = 1;
  }
}

void swap(int** buffer1, int** buffer2){
  int* tmp = *buffer1;
  *buffer1 = *buffer2;
  *buffer2 = tmp;
}

int main(int argc, char * argv[]){

  unsigned int N = argc<2 ? 10 : atoi(argv[1]);

  unsigned int SIZE = N*N; // total size of squared matrix
  int rank, npes;

  MPI_Init( &argc, &argv );

  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // rank store the MPI identifier of the process
  MPI_Comm_size( MPI_COMM_WORLD, &npes ); // npes store the number of MPI processes

  // each process calculates its local nuber of elements
  unsigned int local_n = SIZE/npes;
  int rest = SIZE%npes;
  unsigned int current_n = local_n + (rank<rest); // local nuber of cells to initialize
  
  int* local_m = (int*) calloc(current_n, sizeof(int));
  initLocalMatrix(local_m, rank, current_n, N, npes);

  MPI_Request request;
  MPI_Status status;
  
  if(N<=10){

    // if the process is not the root it just sends  the local matrix initialized
    if (rank) MPI_Send(local_m, current_n, MPI_INT, 0, 101, MPI_COMM_WORLD);

    else{ // the root receives and prints the local matrix as it arrives without storing it

      int* buffer1 = local_m;
      int* buffer2 = (int*) calloc(current_n, sizeof(int));

      int* receiving_buffer;
      int* writing_buffer;

      receiving_buffer = buffer2;
      writing_buffer = buffer1;

      MPI_Irecv(receiving_buffer, local_n + (1<rest), MPI_INT, 1, 101, MPI_COMM_WORLD, &request);
      printPartialMatrix(writing_buffer, N, rank, npes);
      MPI_Wait(&request, &status);

      for (int i = 1; i < npes-1; i++) {

        swap(&receiving_buffer, &writing_buffer);
        current_n = local_n+((i+1)<rest);

        MPI_Irecv(receiving_buffer, current_n, MPI_INT, i+1, 101, MPI_COMM_WORLD, &request);
        printPartialMatrix(writing_buffer, N, i, npes);
        MPI_Wait(&request, &status);

      }
      printPartialMatrix(receiving_buffer, N, npes-1, npes);

    }
  }//end size <=10

  else{//size >10

    if (rank) MPI_Send(local_m, local_n, MPI_INT, 0, 101, MPI_COMM_WORLD);

    else{ // the root process stores the matrix in a binary file as it arrives

      FILE* m_file; char* fileName = "ex4_mtx.dat";
      m_file = fopen(fileName,"wb");

      int previous_n;

      int* buffer1 = local_m;
      int* buffer2 = (int*) calloc(current_n, sizeof(int));

      int* receiving_buffer;
      int* writing_buffer;

      receiving_buffer = buffer2;
      writing_buffer = buffer1;

      MPI_Irecv(receiving_buffer, local_n+(1<rest), MPI_INT, 1, 101, MPI_COMM_WORLD, &request);
      fwrite(writing_buffer, sizeof(int), current_n, m_file);
      MPI_Wait(&request, &status);

      for (int i = 1; i < npes-1; i++) {
          swap(&receiving_buffer, &writing_buffer);
          current_n = local_n + ((i+1)<rest);
          previous_n = local_n + (i<rest);

          MPI_Irecv(receiving_buffer, current_n, MPI_INT, i+1, 101, MPI_COMM_WORLD, &request);
          fwrite(writing_buffer, sizeof(int), previous_n, m_file);
          MPI_Wait(&request, &status);
      }

      fwrite(receiving_buffer, sizeof(int), current_n, m_file);

      free(buffer1); free(buffer2);
      fclose(m_file);
      
      checkMatrix(N, fileName); // check the size of the matrix
    }
  }

  free(local_m);
  MPI_Finalize();

  return 0;
} //end main