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

    __shared__ int A_T[TILED][TILED];
    __shared__ int B_T[TILED][TILED]; 

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < OUT_ROW && col < OUT_COL){
        long long unsigned int val = 0;
        for(int t = 0; t < A_COL / TILED; t++){
            A_T[threadIdx.y][threadIdx.x] = A[row * A_COL + t * TILED + threadIdx.x];
            B_T[threadIdx.y][threadIdx.x] = B[(t * TILED + threadIdx.y) * B_COL + col];
            __syncthreads();
            for(unsigned int k = 0; k < TILED; k++){
                val += A_T[threadIdx.y][k] * B_T[k][threadIdx.x];
            }
            __syncthreads();
        }
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
        ((OUT_ROW-1) / block.y) + 1, 
        ((OUT_COL-1) / block.x) + 1
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
}

#endif