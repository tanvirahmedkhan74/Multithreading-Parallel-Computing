#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel 
{
    int ID = omp_get_thread_num();
    printf("hello(%d) ", ID);
    printf(" world(%d)\n", ID);
}
}

// Run using
// gcc -fopenmp <filename.c> -o <output>
// export OMP_NUM_THREADS=4