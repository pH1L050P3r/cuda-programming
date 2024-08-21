#include <iostream>
#include <chrono>
#include "./generator/generate.hpp"
#include "./CPU/thread.hpp"
#include "./GPU/naive.h"

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()


void print_mat(int R, int C, int* out){
    for(unsigned int i = 0; i < R; i++){
        for(unsigned int j = 0; j < C; j++){
            std::cout << out[i * C + j] << ", "; 
        }
        std::cout << std::endl;
    }
}

int main(){
    unsigned int A_ROW = 10;
    unsigned int A_COL = 10;

    unsigned int B_ROW = 10;
    unsigned int B_COL = 10;

    unsigned int OUT_ROW = A_ROW;
    unsigned int OUT_COL = B_COL;

    int* A = (int *)malloc(A_ROW * A_COL * sizeof(int));
    int* B = (int *)malloc(B_ROW * B_COL * sizeof(int));
    long long unsigned int* OUT_CPU = (long long unsigned int *)malloc(OUT_ROW * OUT_COL * sizeof(long long unsigned int));
    long long unsigned int* OUT_GPU = (long long unsigned int *)malloc(OUT_ROW * OUT_COL * sizeof(long long unsigned int));

    init_matrix(A_ROW, A_COL, A);
    init_matrix(B_ROW, B_COL, B);

    auto begin = TIME_NOW;  
    multiply_matrix_gpu(A_ROW, A_COL, A, B_ROW, B_COL, B, OUT_ROW, OUT_COL, OUT_GPU);
    multiply_matrix_cpu(A_ROW, A_COL, A, B_ROW, B_COL, B, OUT_ROW, OUT_COL, OUT_CPU);
    auto end = TIME_NOW;
    std::cout << "Reference execution time: " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";    
    check_matrix(OUT_ROW, OUT_COL, OUT_CPU, OUT_GPU);
}
