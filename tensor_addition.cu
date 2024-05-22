#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <benchmark/benchmark.h>


__global__ void gpu_addition(int *a, int *b, int *c){
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void cpu_addition(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

static void BM_cpu_addition(benchmark::State& state) {
    int n = 10'000;

    int a[n];
    int b[n];
    int c[n];
    std::fill_n(a, n, 1);
    std::fill_n(b, n, 2);
    std::fill_n(c, n, 0);
    
    for (auto _ : state) {
        cpu_addition(a, b, c, n);
    }
}

static void BM_gpu_addition(benchmark::State& state) {
    int n = 10'000;

    // allocate memory on cpu
    int a[n];
    int b[n];
    int c[n];
    std::fill_n(a, n, 1);
    std::fill_n(b, n, 2);
    std::fill_n(c, n, 0);

    // allocate memory on gpu
    int *cuda_a = 0;
    int *cuda_b = 0;
    int *cuda_c = 0;
    assert(cudaMalloc(&cuda_a, sizeof(a)) == cudaSuccess);
    assert(cudaMalloc(&cuda_b, sizeof(b)) == cudaSuccess);
    assert(cudaMalloc(&cuda_c, sizeof(c)) == cudaSuccess);
    assert(cudaMemcpy(cuda_a, a, sizeof(a), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(cuda_b, b, sizeof(b), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(cuda_c, c, sizeof(c), cudaMemcpyHostToDevice) == cudaSuccess);
    
    // set execution configuration
    int grid_size = 1;  // how many threads per block
    int block_size = n;  // how many blocks

    // TODO: interestingly, it seems like allocating a small block size < 10'000
    // causes the function to run *much* slower. I'm not sure why, will have to
    // do more investigation

    for (auto _ : state) {
        // call function
        gpu_addition <<< grid_size, block_size >>> (cuda_a, cuda_b, cuda_c);
        cudaDeviceSynchronize();
    }

    // copy back to host
    assert(cudaMemcpy(c, cuda_c, sizeof(c), cudaMemcpyDeviceToHost) == cudaSuccess);
}


/*
int main() {
    int a[] = {1, 2, 3};
    int b[] = {5, 6, 7};
    int c[3] = { 0 };

    int *cuda_a = 0;
    int *cuda_b = 0;
    int *cuda_c = 0;

    return 0;
}
*/

BENCHMARK(BM_cpu_addition);
BENCHMARK(BM_gpu_addition);
BENCHMARK_MAIN();