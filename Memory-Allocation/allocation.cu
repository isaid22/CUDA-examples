#include <cuda_runtime.h>
#include <iostream>

__global__ void printAddresses(int *d_data) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Thread %d: Address of d_data[%d] = %p, Value = %d\n", idx, idx, &d_data[idx], d_data[idx]);
}

int main() {
    int N = 10;
    int *h_data = new int[N];
    int *d_data;

    cudaMalloc(&d_data, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_data[i] = i * 10;
    }

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    printAddresses<<<1, N>>>(d_data);

    cudaDeviceSynchronize(); // Wait for the kernel to finish
    cudaFree(d_data);
    delete[] h_data;
    return 0;
}