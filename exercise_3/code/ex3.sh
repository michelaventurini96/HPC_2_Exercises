
module load openmpi
#compile and run the application
mpicc ex_3.c -o ex3

for i in 1 4 8 16 20 32 40; do
    mpirun -np $i ex3 >> results.txt
done

rm ex3
