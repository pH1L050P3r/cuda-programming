#ifndef THREAD_GPU_NAIVE
#define THREAD_CPU_NAIVE

#include "cuda.h"
#include <iostream>

#define THREAD_X 16
#define THREAD_Y 16

__global__ void multiply_mat_n(
    unsigned int A_ROW,
    unsigned int A_COL,
    int* A,
    unsigned int B_ROW,
    unsigned int B_COL,
    int* B,
    unsigned int OUT_ROW,
    unsigned int OUT_COL,
    long long unsigned int* out
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < OUT_ROW && col < OUT_COL){
        long long unsigned int val = 0;
        for(unsigned int k = 0; k < A_COL; k++){
            val += A[row * A_COL + k] * B[k*B_COL + col];
        }
        out[row * OUT_COL + col] = val;
    }
}

void multiply_matrix_gpu_naive(
    unsigned int A_ROW,
    unsigned int A_COL,
    int* A,
    unsigned int B_ROW,
    unsigned int B_COL,
    int* B,
    unsigned int OUT_ROW,
    unsigned int OUT_COL,
    long long unsigned int* OUT
){
    dim3 block(THREAD_Y, THREAD_X);
    dim3 grid(
        (OUT_ROW + block.x - 1) / block.x, 
        (OUT_COL + block.y - 1) / block.y
    );

    int* G_A, *G_B;
    long long unsigned int *G_OUT;

    if(cudaMalloc(&G_A, A_ROW * A_COL * sizeof(int)) != cudaSuccess){
        std::cout << "Unable to allocate Memory GPU : error code = " << cudaGetLastError() << std::endl;
        return;
    }
    if(cudaMalloc(&G_B, B_ROW * B_COL * sizeof(int)) != cudaSuccess){
        std::cout << "Unable to allocate Memory GPU : error code = " << cudaGetLastError() << std::endl;
        return;
    }
    if(cudaMalloc(&G_OUT, OUT_ROW * OUT_COL * sizeof(long long unsigned int)) != cudaSuccess){
        std::cout << "Unable to allocate Memory GPU : error code = " << cudaGetLastError() << std::endl;
        return;
    }

    if(cudaMemcpy(G_A, A, sizeof(int) * A_ROW * A_COL , cudaMemcpyHostToDevice) != cudaSuccess){
        std::cout << "Unable to Copy Memory from CPU to GPU : error code = " << cudaGetLastError() << std::endl;
        return;
    }
    if(cudaMemcpy(G_B, B, sizeof(int) * B_ROW * B_COL , cudaMemcpyHostToDevice) != cudaSuccess){
        std::cout << "Unable to Copy Memory from CPU to GPU : error code = " << cudaGetLastError() << std::endl;
        return;
    }

    multiply_mat_n<<<grid, block>>>(A_ROW, A_COL, G_A, B_ROW, B_COL, G_B, OUT_ROW, OUT_COL, G_OUT);
    
    if(cudaDeviceSynchronize() != cudaSuccess){
        std::cout << "Error while executing above kernel : error code = " << cudaGetLastError() << std::endl;
    }

    if(cudaMemcpy(OUT, G_OUT, sizeof(long long unsigned int) * OUT_ROW * OUT_COL, cudaMemcpyDeviceToHost) != cudaSuccess){
        std::cout << "Unable to Copy Memory from GPU to CPU : error code = " << cudaGetLastError() << std::endl;
        return;
    }
    cudaFree(G_A);
    cudaFree(G_B);
    cudaFree(G_OUT);
}

#endif