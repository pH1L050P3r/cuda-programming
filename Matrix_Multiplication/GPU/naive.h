#ifndef THREAD_GPU_NAIVE
#define THREAD_CPU_NAIVE

#include "cuda.h"
#include <iostream>

#define THREAD_X 32
#define THREAD_Y 32

__global__ void multiply_mat_k(
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
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < OUT_ROW && col < OUT_COL){
        long long unsigned int val = 0;
        for(unsigned int k = 0; k < A_COL; k++){
            val += A[row * A_COL + k] * B[k*B_COL + col];
        }
        out[row * OUT_COL + col] = val;
    }
}

void multiply_matrix_gpu(
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
        ((OUT_ROW-1) / block.y) + 1, 
        ((OUT_COL-1) / block.x) + 1
    );

    int* G_A, *G_B;
    long long unsigned int *G_OUT;

    if(cudaMalloc(&G_A, A_ROW * A_COL * sizeof(int)) != cudaSuccess){
        std::cout << "Unable to allocate Memory GPU" << std::endl;
        return;
    }
    cudaMalloc(&G_B, B_ROW * B_COL * sizeof(int));
    cudaMalloc(&G_OUT, OUT_ROW * OUT_COL * sizeof(long long unsigned int));

    cudaMemcpy(G_A, A, sizeof(int) * A_ROW * A_COL , cudaMemcpyHostToDevice);
    cudaMemcpy(G_B, B, sizeof(int) * B_ROW * B_COL , cudaMemcpyHostToDevice);

    multiply_mat_k<<<grid, block>>>(A_ROW, A_COL, A, B_ROW, B_COL, B, OUT_ROW, OUT_COL, OUT);

    cudaMemcpy(OUT, G_OUT, sizeof(long long unsigned int) * OUT_ROW * OUT_COL, cudaMemcpyDeviceToHost);

    std::cout << OUT[0] << std::endl;
}

#endif