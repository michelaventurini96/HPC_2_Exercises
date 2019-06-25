
gcc -fopenmp ex_2.c -o ex_2

export OMP_NUM_THREADS=4
echo "Number of threads = ${OMP_NUM_THREADS}"
./ex_2 >> results.txt

rm ex_2
