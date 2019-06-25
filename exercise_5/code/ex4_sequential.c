#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char * argv[])
{
  int rank, npes;
  double startwtime, endwtime;
  MPI_Init( &argc, &argv );

  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // rank store the MPI identifier of the process
  MPI_Comm_size( MPI_COMM_WORLD, &npes ); // npes store the number of MPI processes

  //Synchronize all processes and get the begin time
  MPI_Barrier(MPI_COMM_WORLD);
  startwtime = MPI_Wtime();

  int local_buf = rank;
  int rec_buf = 0;

  int right = (rank + 1) % npes;
  int left = rank - 1;
  if (left < 0) left = npes - 1;

  for (int i = 0; i < npes-1; i++) {
    MPI_Send(&local_buf, 1, MPI_INT, left, 101, MPI_COMM_WORLD);
    MPI_Recv(&rec_buf, 1, MPI_INT, right, 101, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
   
    local_buf=rec_buf+rank;
  }

  //Synchronize all processes and get the end time
  MPI_Barrier(MPI_COMM_WORLD);
  endwtime = MPI_Wtime();

  int expected = (npes)*(npes-1)/2;
  if (rank==0) printf("Time of execution: %f\n", endwtime-startwtime);

  printf("Rank=%d, Sum = %d (Should be %d)\n", rank, local_buf, expected);

  MPI_Finalize();



  return 0;
} //end main
