!/bin/bash
#run the application
gcc -fopenmp ex_1.c -Wall -Wextra -o ex1

for i in 1 4 8 16 20; do
  OMP_NUM_THREADS=$i ./ex1_c >> results.txt
done
