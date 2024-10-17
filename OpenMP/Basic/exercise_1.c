#include<stdio.h>
#include<omp.h>
#include <stdlib.h>

#define MAX_THREAD 10

static long num_steps = 100000;
double step;

int main() {
    
    double pi, sum = 0.0;
    double *local_sums = (double *) malloc(12 * sizeof(double));

    step = 1.0/ (double)num_steps;

    for(int i = 0; i < 12; i++) {
        local_sums[i] = 0.0;
    }

    double start_time = omp_get_wtime();

    omp_set_num_threads(MAX_THREAD);
    #pragma omp parallel
    {
        int i;
        int num_threads = omp_get_num_threads();
        printf("Num of threads = %d\n", num_threads);
        int tid, start, end;
        double x, local_sum;

        tid = omp_get_thread_num();

        start = tid * (num_steps/num_threads);
        end   = (tid + 1) * (num_steps/num_threads);

        for(i=start;i<end;i++){
            x = (i+0.5)*step;
            local_sum = 4.0/(1.0 + x*x);
            local_sums[tid] += local_sum;
        }
    }

    for(int i = 0; i < 12; i++) {
        sum += local_sums[i];
        printf("Local sum for thread %d is %f\n", i, local_sums[i]);
    }

    pi = step * sum;
    printf("Pi after integration under the area: %f\n", pi);

    double end_time = omp_get_wtime();
    printf("Time taken by the multi threading: %f\n", end_time - start_time);

    free(local_sums);
}