#ifndef GENERATE_H
#define GENERATE_H

#include <iostream>
#include <stdlib.h>
#include <assert.h>

void init_matrix(int ROW, int COL, int* MAT){
    assert(0 <= ROW);
    assert(0 <= COL);

    for(unsigned int i = 0; i < ROW; i++){
        for(unsigned int j = 0; j < COL; j++){
            MAT[i * COL + j] = rand() % 1000;
        }
    }
}

void check_matrix(int ROW, int COL, long long unsigned int* SRC, long long unsigned int* ANS){
    for(unsigned int i = 0; i < ROW; i++){
        for(unsigned int j = 0; j < COL; j++){
            // std::cout << (SRC[i*COL + j] == ANS[i*COL + j]) << std::endl;
            // std::cout << SRC[i*COL + j] << " " << ANS[i*COL + j] << " " << i << " " << j << std::endl;
            assert(SRC[i*COL + j] == ANS[i*COL + j]);
        }
    }
}


void GPU_Warmup(){
    int* A;
    cudaMalloc(&A, sizeof(int) * 10240);
    cudaFree(A);
}

#endif