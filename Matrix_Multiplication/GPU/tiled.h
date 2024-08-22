#ifndef THREAD_GPU_TILED
#define THREAD_CPU_TILED

#include "cuda.h"
#include <iostream>

#define THREAD_X 16
#define THREAD_Y 16
#define TILED  16

__global__ void multiply_mat_t(
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

    long long unsigned int val = 0;
    for(int t = 0; t < A_COL ; t += TILED){
        __shared__ int A_T[TILED][TILED];
        __shared__ int B_T[TILED][TILED]; 
        
        if(t + threadIdx.y < A_COL) 
            A_T[threadIdx.x][threadIdx.y] = A[row * A_COL + t + threadIdx.y];
        else 
            A_T[threadIdx.x][threadIdx.y] = 0;
        
        if(t + threadIdx.x < B_ROW)
            B_T[threadIdx.x][threadIdx.y] = B[(t + threadIdx.x) * B_COL + col];
        else
            B_T[threadIdx.x][threadIdx.y] = 0;

        __syncthreads();
        for(unsigned int k = 0; k < TILED; k++){
            val += A_T[threadIdx.x][k] * B_T[k][threadIdx.y];
        }
        __syncthreads();
    }
    if(row < OUT_ROW && col < OUT_COL){
        out[row * OUT_COL + col] = val;
    }
}

void multiply_matrix_gpu_tiled(
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

    multiply_mat_t<<<grid, block>>>(A_ROW, A_COL, G_A, B_ROW, B_COL, G_B, OUT_ROW, OUT_COL, G_OUT);
    
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