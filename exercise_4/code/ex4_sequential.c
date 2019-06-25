#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

size_t globalIdx(const unsigned int n, const int npes, const int rank){
  int elems = (n*n)/npes;
  int rest = (n*n)%npes;
  
  size_t global_i = 0;

  for(int i=0; i<rank; i++)
    global_i += (elems + (i<rest));

  return global_i;
}

void printPartialMatrix(const int* const matrix, const unsigned int n, const int rank, const int npes){
  size_t global_i = globalIdx(n, npes, rank);
  int rest = (n*n)%npes, elems = n*n/npes + (rank<rest);

  for (int i = 0; i < elems; i++) {
      printf("%d ", matrix[i]);
      int res=(global_i+1)%n;
      if( res==0 && global_i!=0 ) printf("\n");
      global_i++;
  }
}

void printAllMatrix(const int* const matrix, const unsigned int n){
  for (unsigned int i = 0; i < (n*n); i++) {
      printf("%d ", matrix[i]);
      int res = (i+1)%n;
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

void initLocalMatrix(int *m, const int rank, const unsigned int local_elems, const unsigned int n, int npes){
  size_t global_i = globalIdx(n, npes, rank);

  for (unsigned int i = 0; i < local_elems; i++){
    if (i+global_i==0 || (i+global_i)%(n+1)==0) m[i] = 1;
  }
}

int main(int argc, char * argv[]){

  size_t N = argc<2 ? 10 : atoi(argv[1]);

  unsigned int SIZE = N*N;
  int rank, npes;

  MPI_Init( &argc, &argv );

  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // rank store the MPI identifier of the process
  MPI_Comm_size( MPI_COMM_WORLD, &npes ); // npes store the number of MPI processes

  //each process calculates its local nuber of elements
  int rest=SIZE%npes;
  unsigned int local_n = (SIZE/npes) + (rank<rest);

  //initialization of local matrix
  int* local_m = (int*) calloc(local_n, sizeof(int));
  initLocalMatrix(local_m, rank, local_n, N, npes);

  if(N<=10){
    if (rank) MPI_Send(local_m, local_n, MPI_INT, 0, 101, MPI_COMM_WORLD);
    else{
      printPartialMatrix(local_m, N, rank, npes);

      for (int i = 1; i < npes; i++) {
        MPI_Recv(local_m, local_n, MPI_INT, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printPartialMatrix(local_m, N, i, npes);
      }
    }
  }//end size <=10

  else{//size >10
    if (rank) MPI_Send(local_m, local_n, MPI_INT, 0, 101, MPI_COMM_WORLD);
    else{
      FILE* m_file; char* fileName = "ex4_mtx.dat";
      m_file = fopen(fileName,"wb");

      int *buffer = (int*) calloc(local_n, sizeof(int)); 
      int local_n = SIZE/npes;
      int current_n = local_n + (rank<rest);

      fwrite(local_m, sizeof(int), current_n, m_file);

      for (int i = 1; i < npes; i++) {
          current_n = local_n + (i<rest);
          MPI_Recv(buffer, current_n, MPI_INT, i, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          fwrite(buffer, sizeof(int), current_n, m_file);
      }
      free(buffer);
      fclose(m_file);

      checkMatrix(N, fileName);
    }
  }

  free(local_m);
  MPI_Finalize();

  return 0;
} //end main
