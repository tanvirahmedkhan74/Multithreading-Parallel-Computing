/**
 * @file pi_parallel_integration.c
 * @brief Parallel computation of the value of Pi using numerical integration and OpenMP.
 *
 * This program calculates the value of Pi by integrating the function 4 / (1 + x^2) over the range [0, 1]
 * using the midpoint rule for numerical integration. The computation is parallelized using OpenMP to
 * divide the workload among multiple threads.
 * 
 * Key concepts in the code:
 * - Numerical integration using the midpoint rule.
 * - Parallelism via OpenMP with thread-level private and shared variables.
 * - Dynamic memory allocation using `malloc` for storing partial sums of each thread on Heap.
 * - Threads divide the work by calculating a portion of the integral, and their results are combined at the end.
 * - Thread-local sums are accumulated and then summed together to get the final result.
 *
 * The number of threads can be controlled via the `MAX_THREAD` macro. By default, the code runs with 10 threads.
 * The value of Pi computed will be printed along with the time taken for the computation.
 *
 * @author Tanvir Ahmed Khan
 * @date 10/18/2024
 *
 * @compile gcc -fopenmp pi_parallel_integration.c -o pi_parallel_integration
 * @run ./pi_parallel_integration
 *
 * @params:
 * - MAX_THREAD: The maximum number of threads used for parallel computation (set to 10 by default).
 * - num_steps: The number of steps (rectangles) used for integration (set to 100,000).
 *
 * @outputs:
 * - The computed value of Pi after integration.
 * - The local sum for each thread.
 * - The total time taken for the parallel computation.
 */

#include<stdio.h>
#include<omp.h>
#include <stdlib.h>

#define MAX_THREAD 10

static long num_steps = 100000;
double step;

int main() {
    
    double pi, sum = 0.0;
    double *local_sums = (double *) malloc(12 * sizeof(double));

    // Step width for midpoint rule for numerical integraion
    step = 1.0/ (double)num_steps;

    for(int i = 0; i < 12; i++) {
        local_sums[i] = 0.0;
    }

    double start_time = omp_get_wtime();

    omp_set_num_threads(MAX_THREAD);
    #pragma omp parallel
    {
        // Private Variables for each thread
        int i, num_threads, tid, start, end;
        double x, local_sum;

        num_threads = omp_get_num_threads();
        printf("Num of threads = %d\n", num_threads);

        tid = omp_get_thread_num();

        // Each Thread works on a portion of the iteration
        start = tid * (num_steps/num_threads);
        end   = (tid + 1) * (num_steps/num_threads);

        for(i=start;i<end;i++){
            x = (i+0.5)*step;
            local_sum = 4.0/(1.0 + x*x);
            local_sums[tid] += local_sum;
        }
    }

    // Summing up the sums of each thread
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