#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;

cudaGetDeviceProperties(&prop, 0); // Query device 0 or default GPU

printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
printf("Shared memory per multiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);

return 0;

}

/*
Shared memory per block: 49152 bytes
Shared memory per multiprocessor: 102400 bytes
*/