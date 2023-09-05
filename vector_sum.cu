#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The size of the vector
#define NUM_DATA 1024 * 1024

// Simple vector sum kernel
__global__ void vecAdd(int* _a, int* _b, int* _c, int _size) {
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tID < _size)
        _c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
    int* a,  * b, * c, * hc;    // Vectors on the host
    int* da, * db, * dc;        // Vector on the device

    dim3 dimGrid(ceil((float)NUM_DATA / 256), 1, 1);
    dim3 dimBlock(256, 1, 1);

    int memSize = sizeof(int) * NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    DS_timer timer(5);
    timer.initTimers();

    // Memory allocation on the host-side
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    hc = new int[NUM_DATA]; memset(hc, 0, memSize);

    // Data generation
    for (int i = 0; i < NUM_DATA; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }

    // Vector sum on host (for performance comparsion)
    timer.onTimer(0);
    for (int i = 0; i < NUM_DATA; i++)
        hc[i] = a[i] + b[i];
    timer.offTimer(0);

    // Memory allocation on the device-side
    cudaMalloc(&da, memSize); cudaMemset(da, 0, memSize);
    cudaMalloc(&db, memSize); cudaMemset(db, 0, memSize);
    cudaMalloc(&dc, memSize); cudaMemset(dc, 0, memSize);
    
    timer.onTimer(2);
    // Data copy : Host -> Device
    cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    // Kernel call
    timer.onTimer(1);
    vecAdd <<<dimGrid, dimBlock>>> (da, db, dc, NUM_DATA);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    // Copy results: Device -> Host
    cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(da); cudaFree(db); cudaFree(dc);

    // Check results
    bool result = true;
    for (int i = 0; i < NUM_DATA ; i++) {
        if (hc[i] != c[i]) {
            printf("[%d] The result is not matched! (%d, %d)\n", i, hc[i], c[i]);
            result = false;
        }
    }

    if (result)
        printf("GPU works well!\n");

    timer.printTimer();

    // Release host memory
    delete[] a; delete[] b; delete[] c;

    return 0;
}