#include <stdio.h>
#include <assert.h>

__global__ void vector_add(int* a, int* b, int* c, int n){
    // calc global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n){
        c[tid] = a[tid] + b[tid];
    }
}

// init vector of size n, with values between 0 and 99
void maxtrix_init(int* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
    }
}

void error_check(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main(int argc, char** argv){
    
    int n = 1 << 16; // vector size of 2^16

    int *h_a, *h_b, *h_c; // host vector pointers
    int *d_a, *d_b, *d_c; // device vector pointers
    size_t bytes = sizeof(int) * n; // allocation size for vectors

    // allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // initialize vectors a and b
    maxtrix_init(h_a, n);
    maxtrix_init(h_b, n);

    // copy vectors a and b to device memory (gpu) from host memory (cpu)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice); // inside of the device memory
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    // pointer (d_a) put content of h_a of size bytes

    // Thread block size
    int NUM_THREADS = 512;

    // Grid size
    int NUM_BLOCKS = (int) ceil(n / NUM_THREADS); // we want a single thread calculating each element of vector addition

    // launch kernel
    vector_add<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // copy results from device memory (gpu) to host memory (cpu)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // error checking
    error_check(h_a, h_b, h_c, n);

    // free memory
    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("DONE!\n");
    return 0;
}