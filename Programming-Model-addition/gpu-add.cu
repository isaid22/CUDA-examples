# include <stdio.h>
# include "../common/book.h"
#define N   10

__global__ void add( int *a, int *b, int *c ) {
    int tid = blockIdx.x;    // this thread handles the data at its thread id
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}





int main( void ) {
    int a[N], b[N], c[N]; // Each array holds N integers
    int *dev_a, *dev_b, *dev_c; //pointers declared to point to integer array's base address

    
    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( &dev_a, N * sizeof(int) ) ); // Allocate continuous block memory, integer is four bytes.
    HANDLE_ERROR( cudaMalloc( &dev_b, N * sizeof(int) ) ); // Pointer to a pointer indicates address of the allocated memory on the device. 
    HANDLE_ERROR( cudaMalloc( &dev_c, N * sizeof(int) ) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
                              cudaMemcpyHostToDevice ) );

    add<<<N,1>>>( dev_a, dev_b, dev_c );

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
                              cudaMemcpyDeviceToHost ) );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // free the memory allocated on the GPU
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaFree( dev_b ) );
    HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}
