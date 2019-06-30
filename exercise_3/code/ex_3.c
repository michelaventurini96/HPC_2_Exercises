#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double func (double x){
  return (1/(1+(x*x)));
}

double local_sum(double local_a, double local_b, double h){
  double midpoint = local_a+h/2;
  double local_res = 0;
  while (midpoint<local_b) {
    midpoint +=h;
    local_res += h*func(midpoint);
  }
  return local_res;
}

int main(int argc, char * argv[])
{

  int n = 100000000, rank, npes, error;
  double  a=0, b=1, global_res=0, h = (b-a)/n, pi=0,
          startwtime=0, endwtime=0;

  error = MPI_Init( &argc, &argv );

  MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // rank store the MPI identifier of the process
  MPI_Comm_size( MPI_COMM_WORLD, &npes ); // npes store the number of MPI processes

  //Synchronize all processes and get the begin time
  MPI_Barrier(MPI_COMM_WORLD);
  startwtime = MPI_Wtime();

  //each process calculates its local sum
  int local_n = n/npes;
  double local_a = a+rank*local_n*h;
  double local_b = local_a + local_n*h;
  double local_res = local_sum(local_a, local_b, h);

  //Consolidate and Sum Results in the proc npes-1
  MPI_Reduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_SUM, npes-1, MPI_COMM_WORLD);

  //the last process send the result to the first one
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank==npes-1) {
    pi = 4*global_res;
    MPI_Send(&pi, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
  }

  //Synchronize all processes and get the end time
  MPI_Barrier(MPI_COMM_WORLD);
  endwtime = MPI_Wtime();

  //the process with rank zero print the result and the time of execution
  if (!rank){
    MPI_Recv(&pi, 1, MPI_DOUBLE, npes-1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("pi = %f\n", pi);
    printf("Time of execution: %f\n", endwtime-startwtime);

  }
  error=MPI_Finalize();
  return 0;
}
