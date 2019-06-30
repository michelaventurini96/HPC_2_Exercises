#include<stdlib.h>
#include <stdio.h>
#include <omp.h>

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

int main()
{
  int n = 100000000;
  double  a=0, b=1;
  double global_res = 0;
  double h = (b-a)/n;

  //SERIAL
  printf("Serial execution\n");
  double start = omp_get_wtime();
  double pi  = local_sum(a, b, h);
  double end = omp_get_wtime();
  printf("pi = %f, time = %f \n",pi, end-start);

  // ATOMIC
  printf("Atomic execution\n");
  global_res = 0;
  start=omp_get_wtime();
  #pragma omp parallel
  {
   
    int t_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    //printf("%d\n", n_threads);

    int local_n = n/n_threads;
    double local_a = a+t_id*local_n*h;
    double local_b = local_a + local_n*h;
    double local_s = local_sum(local_a, local_b, h);
    #pragma omp atomic
    global_res += local_s;
  }

  end = omp_get_wtime();
  pi = 4*global_res;
  printf("pi= %f, with time %f s\n", pi, end-start);

  //CRITICAL
  printf("Critical execution\n");
  global_res = 0;
  start=omp_get_wtime();
  #pragma omp parallel
  {
    int t_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    //printf("%d\n", n_threads);

    int local_n = n/n_threads;
    double local_a = a+t_id*local_n*h;
    double local_b = local_a + local_n*h;
    double local_s = local_sum(local_a, local_b, h);
    #pragma omp critical 
    {
      global_res += local_s;
    }
  }

  end = omp_get_wtime();
  pi = 4*global_res;
  printf("pi= %f, with time %f s\n", pi, end-start);


  //REDUCTION
  printf("Reduction execution\n");
  global_res = 0;
  start=omp_get_wtime();
  #pragma omp parallel
  {
   
    int t_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    //printf("%d\n", n_threads);

    int local_n = n/n_threads;
    double local_a = a+t_id*local_n*h;
    double local_b = local_a + local_n*h;

    #pragma omp parallel reduction(+:global_res)
    	global_res += local_sum(local_a, local_b, h);
  }

  end = omp_get_wtime();
  pi = 4*global_res;
  printf("pi= %f, with time %f s\n", pi, end-start);

}
