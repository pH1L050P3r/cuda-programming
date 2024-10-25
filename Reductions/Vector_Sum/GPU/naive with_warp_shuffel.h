#ifndef VECTOR_SUM_GPU_H
#define VECTOR_SUM_GPU_H

#include <iostream>
#include <vector>
#include "cuda.h"

namespace std
{


#define BLOCK_SIZE 1024
const int mod = (1 << 30);

__global__ void vector_sum_k(int* arr, unsigned int size, int *out){
    __shared__  int block[BLOCK_SIZE];
    
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < size)
        block[threadIdx.x] = arr[id];
    else block[threadIdx.x] = 0;

    __syncthreads();
    for(unsigned int stride  = blockDim.x / 2; stride >= warpSize; stride = stride >> 1){
        if(threadIdx.x < stride) block[threadIdx.x] = ( block[threadIdx.x] + block[threadIdx.x + stride]) % mod;
        __syncthreads();
    }

    // Warp reduction
    int sum;
    if(threadIdx.x < warpSize){
        sum = block[threadIdx.x];
        for(unsigned int stride = warpSize / 2; stride > 0; stride /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        }
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}


void vector_sum_gpu(int* vec, size_t size, int *total){
    int *g_in;
    int *g_out, *out;

    const unsigned block = BLOCK_SIZE;
    const unsigned grid = ((size - 1) / block) + 1;

    if(cudaMalloc(&g_in, sizeof(int) * size) != cudaSuccess){
        std::cout << "Error while alocating cuda Memory. -1" << cudaGetLastError() << std::endl;
        exit(-1);
    }

    if(cudaMalloc(&g_out, sizeof(int) * grid) != cudaSuccess){
        std::cout << "Error while alocationg cuda Memory. -2" << cudaGetLastError() << std::endl;
        exit(-2);
    }
    out = (int *)malloc(grid * sizeof(int));

    if(cudaMemcpy(g_in, vec, sizeof(int)*size, cudaMemcpyHostToDevice) != cudaSuccess){
        std::cout << "Error while copying memory. -3" << cudaGetLastError() << std::endl;
        exit(-3);
    }

    if(cudaMemset(g_out, 0, sizeof(int)) != cudaSuccess){
        std::cout << "Error while copying memory. -4" << cudaGetLastError() << std::endl;
        exit(-3);
    }

    
    vector_sum_k<<<grid, block>>>(g_in, size, g_out);
    if(cudaMemcpy(out, g_out, sizeof(int) * grid, cudaMemcpyDeviceToHost) != cudaSuccess){
        std::cout << "Error while copying Memory. -5" << cudaGetLastError() << std::endl;
        exit(-4);
    }

    int t_out = 0;
    for(int i = 0; i < grid; i++){
        t_out = (t_out + out[i]) % mod;
    }

    *total = t_out;

    cudaFree(g_in);
    cudaFree(g_out);
}

};
#endif