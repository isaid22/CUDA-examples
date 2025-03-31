#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dot(float *a, float *b, float *result, int N) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        atomicAdd(result, cache[0]);
}

int main() {
    int N = 1024;
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_result;
    float h_result = 0;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    dot<<<4, 256>>>(d_a, d_b, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot product result: %f\n", h_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    delete[] h_a;
    delete[] h_b;
    return 0;
}
